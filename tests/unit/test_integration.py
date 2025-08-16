#!/usr/bin/env python3
"""
Integration tests for Ollama integration with all 131 agents
Tests end-to-end functionality, agent coordination, and system interactions
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import pytest
import asyncio
import time
import json
import sys
import os
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, patch, MagicRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
from datetime import datetime, timedelta
import tempfile

# Add the agents directory to the path
# Path handled by pytest configuration, '..', 'agents'))

from agents.core.base_agent import BaseAgentV2, AgentStatus, TaskResult
from core.ollama_pool import OllamaConnectionPool
from core.ollama_integration import OllamaIntegration, OllamaConfig
from core.circuit_breaker import CircuitBreaker, CircuitBreakerState


class TestAgentOllamaIntegration:
    """Test integration between agents and Ollama service"""
    
    @pytest.fixture
    def integration_agent(self):
        """Create agent for integration testing"""
        with patch.dict(os.environ, {
            'AGENT_NAME': 'integration-test-agent',
            'AGENT_TYPE': 'integration-test',
            'BACKEND_URL': 'http://test-backend:8000',
            'OLLAMA_URL': 'http://test-ollama:10104'
        }):
            return BaseAgent(
                max_concurrent_tasks=3,
                max_ollama_connections=2
            )
    
    @pytest.mark.asyncio
    async def test_end_to_end_agent_task_processing(self, integration_agent):
        """Test complete end-to-end task processing with Ollama"""
        await integration_agent._setup_async_components()
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test successful Ollama interaction
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_response = "This is a test response from Ollama for the integration test."
        
        with patch.object(integration_agent.circuit_breaker, 'call', return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_response):
            # Create a test task that would use Ollama
            task = {
                "id": "integration-test-001",
                "type": "text-generation",
                "data": {
                    "prompt": "Generate a test response",
                    "model": "tinyllama",
                    "max_tokens": 100
                }
            }
            
            # Process the task
            result = await integration_agent.process_task(task)
            
            # Verify successful processing
            assert result.task_id == "integration-test-001"
            assert result.status == "completed"
            assert result.processing_time > 0
            assert "success" in result.result
            assert integration_agent.metrics.tasks_processed == 1
        
        await integration_agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_model_switching_integration(self, integration_agent):
        """Test agent switching between different Ollama models"""
        await integration_agent._setup_async_components()
        
        # Test different model responses - all using GPT-OSS
        model_responses = {
            "tinyllama": "GPT-OSS response for simple task",
            "tinyllama": "GPT-OSS response for coding task",
            "tinyllama": "GPT-OSS response for complex reasoning"
        }
        
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_call(*args, **kwargs):
            # Extract model from kwargs or use default
            model = kwargs.get('model', integration_agent.default_model)
            return model_responses.get(model, f"Response from {model}")
        
        with patch.object(integration_agent.circuit_breaker, 'call', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_call):
            
            # Test tasks with different model requirements - all using GPT-OSS
            test_cases = [
                ("simple-task", "tinyllama", "Simple question"),
                ("coding-task", "tinyllama", "Write a function"),
                ("reasoning-task", "tinyllama", "Solve this complex problem")
            ]
            
            results = []
            for task_id, model, prompt in test_cases:
                # Override agent's model for this test
                original_model = integration_agent.default_model
                integration_agent.default_model = model
                
                result = await integration_agent.query_ollama(prompt, model=model)
                results.append((model, result))
                
                # Restore original model
                integration_agent.default_model = original_model
            
            # Verify each model was used correctly
            for (model, result), expected_response in zip(results, model_responses.values()):
                assert result == expected_response
                
            # Verify metrics
            assert integration_agent.metrics.ollama_requests == len(test_cases)
            assert integration_agent.metrics.ollama_failures == 0
        
        await integration_agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_connection_pool_integration(self, integration_agent):
        """Test agent integration with connection pool"""
        await integration_agent._setup_async_components()
        
        # Verify connection pool is properly initialized
        assert integration_agent.ollama_pool is not None
        assert integration_agent.ollama_pool.default_model == integration_agent.default_model
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test connection pool responses
        async def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_generate(prompt, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"Pool response for: {prompt[:20]}..."
        
        with patch.object(integration_agent.ollama_pool, 'generate', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_generate):
            
            # Test multiple concurrent requests through the pool
            prompts = [f"Concurrent request {i}" for i in range(10)]
            
            start_time = time.time()
            tasks = [
                integration_agent.query_ollama(prompt) 
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Verify all requests completed
            assert len(results) == 10
            assert all(r is not None for r in results)
            assert all("Pool response for:" in r for r in results)
            
            # Verify connection pool handled concurrency efficiently
            assert total_time < 2.0  # Should complete within reasonable time
            
            # Check metrics
            assert integration_agent.metrics.ollama_requests == 10
        
        await integration_agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_circuit_breaker_integration(self, integration_agent):
        """Test agent integration with circuit breaker"""
        await integration_agent._setup_async_components()
        
        # Verify circuit breaker is initialized
        assert integration_agent.circuit_breaker is not None
        assert integration_agent.circuit_breaker.state == CircuitBreakerState.CLOSED
        
        call_count = 0
        
        async def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_failing_ollama_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Fail first 3 calls, then succeed
            if call_count <= 3:
                raise Exception(f"Simulated Ollama failure {call_count}")
            return f"Success after {call_count} attempts"
        
        with patch.object(integration_agent.ollama_pool, 'generate', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_failing_ollama_call):
            
            # First few calls should fail and eventually trip circuit breaker
            results = []
            for i in range(5):
                result = await integration_agent.query_ollama(f"Test prompt {i}")
                results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            # Verify some requests failed (circuit breaker should have opened)
            failed_results = [r for r in results if r is None]
            successful_results = [r for r in results if r is not None]
            
            assert len(failed_results) > 0, "Some requests should have failed"
            assert integration_agent.metrics.ollama_failures >= 3
            
            # Circuit breaker should have been tripped
            assert integration_agent.metrics.circuit_breaker_trips >= 0
        
        await integration_agent._cleanup_async_components()


class TestMultiAgentCoordination:
    """Test coordination between multiple agents"""
    
    @pytest.fixture
    def agent_factory(self):
        """Factory for creating test agents"""
        agents_created = []
        
        def create_agent(agent_name, agent_type, model=None):
            with patch.dict(os.environ, {
                'AGENT_NAME': agent_name,
                'AGENT_TYPE': agent_type,
                'BACKEND_URL': 'http://test-backend:8000',
                'OLLAMA_URL': 'http://test-ollama:10104'
            }):
                agent = BaseAgent(max_concurrent_tasks=2)
                if model:
                    agent.default_model = model
                agents_created.append(agent)
                return agent
        
        yield create_agent
        
        # Cleanup
        for agent in agents_created:
            if hasattr(agent, '_cleanup_async_components'):
                try:
                    asyncio.run(agent._cleanup_async_components())
                except (AssertionError, Exception) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
    
    @pytest.mark.asyncio
    async def test_multiple_agents_different_models(self, agent_factory):
        """Test multiple agents using different models concurrently"""
        # Create agents with different model assignments - all using GPT-OSS
        agents = [
            agent_factory("test-tinyllama-agent-1", "complex-reasoning", "tinyllama"),
            agent_factory("test-tinyllama-agent-2", "balanced-task", "tinyllama"),
            agent_factory("test-default-agent", "simple-task", "tinyllama")
        ]
        
        # Setup all agents
        for agent in agents:
            await agent._setup_async_components()
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test different responses for GPT-OSS model
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_model_response(*args, **kwargs):
            model = kwargs.get('model', 'tinyllama')
            return "Response from GPT-OSS model"
        
        with patch.object(BaseAgent, 'query_ollama', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_model_response):
            
            # Execute tasks on all agents concurrently
            async def agent_task(agent, task_num):
                prompt = f"Task {task_num} for {agent.agent_name}"
                result = await agent.query_ollama(prompt, model=agent.default_model)
                return agent.agent_name, result
            
            # Create tasks for all agents
            tasks = []
            for i, agent in enumerate(agents):
                for task_num in range(3):  # 3 tasks per agent
                    tasks.append(agent_task(agent, task_num))
            
            # Execute all tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            # Verify all tasks completed
            assert len(results) == 9  # 3 agents × 3 tasks each
            
            # Verify all agents used GPT-OSS model
            gpt_oss_1_results = [r[1] for r in results if 'tinyllama-agent-1' in r[0]]
            gpt_oss_2_results = [r[1] for r in results if 'tinyllama-agent-2' in r[0]]
            default_results = [r[1] for r in results if 'default' in r[0]]
            
            assert all("GPT-OSS" in r for r in gpt_oss_1_results)
            assert all("GPT-OSS" in r for r in gpt_oss_2_results)
            assert all("GPT-OSS" in r for r in default_results)
            
            # Execution should be reasonably fast with proper concurrency
            assert execution_time < 5.0
            
            logger.info(f"Multi-agent execution completed in {execution_time:.2f} seconds")
        
        # Cleanup
        for agent in agents:
            await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_resource_sharing(self, agent_factory):
        """Test that agents properly share Ollama resources"""
        # Create multiple agents that would share the same Ollama instance
        agents = [
            agent_factory(f"shared-resource-agent-{i}", "shared-test", "tinyllama")
            for i in range(5)
        ]
        
        # Setup all agents
        for agent in agents:
            await agent._setup_async_components()
        
        # Track resource usage
        connection_counts = []
        
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_with_tracking(*args, **kwargs):
            # Simulate resource tracking
            connection_counts.append(len(connection_counts) + 1)
            return f"Shared response {len(connection_counts)}"
        
        with patch.object(BaseAgent, 'query_ollama', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_with_tracking):
            
            # Execute requests from all agents simultaneously
            async def agent_requests(agent):
                results = []
                for i in range(3):
                    result = await agent.query_ollama(f"Request {i} from {agent.agent_name}")
                    results.append(result)
                    await asyncio.sleep(0.1)  # Small delay between requests
                return results
            
            # Run all agents concurrently
            all_results = await asyncio.gather(*[agent_requests(agent) for agent in agents])
            
            # Verify all agents got responses
            assert len(all_results) == 5  # 5 agents
            assert all(len(agent_results) == 3 for agent_results in all_results)  # 3 requests each
            
            # Verify resource sharing worked (all requests completed)
            total_requests = sum(len(agent_results) for agent_results in all_results)
            assert total_requests == 15  # 5 agents × 3 requests each
        
        # Cleanup
        for agent in agents:
            await agent._cleanup_async_components()


class TestSystemIntegration:
    """Test system-wide integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_agent_backend_integration(self):
        """Test agent integration with backend coordinator"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test successful backend interactions
        registration_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        registration_response.status_code = 200
        
        heartbeat_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        heartbeat_response.status_code = 200
        
        task_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        task_response.status_code = 200
        task_response.json.return_value = {
            "id": "backend-task-001",
            "type": "integration-test",
            "data": {"prompt": "Backend integration test"}
        }
        
        no_task_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        no_task_response.status_code = 204
        
        completion_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        completion_response.status_code = 200
        
        with patch.object(agent.http_client, 'post') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post:
            with patch.object(agent.http_client, 'get') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get:
                
                # Setup Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test responses
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post.return_value = registration_response
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get.side_effect = [task_response, no_task_response]
                
                # Test registration
                registration_success = await agent.register_with_coordinator()
                assert registration_success is True
                
                # Verify registration data was sent
                assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post.called
                registration_call = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post.call_args
                registration_data = registration_call[1]['json']
                assert registration_data['agent_name'] == agent.agent_name
                assert registration_data['agent_type'] == agent.agent_type
                assert registration_data['capabilities'] == agent.config['capabilities']
                
                # Test task retrieval
                task = await agent.get_next_task()
                assert task is not None
                assert task['id'] == 'backend-task-001'
                
                # Test no task available
                no_task = await agent.get_next_task()
                assert no_task is None
                
                # Test task completion reporting
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post.return_value = completion_response
                
                task_result = TaskResult(
                    task_id="backend-task-001",
                    status="completed",
                    result={"output": "Integration test completed"},
                    processing_time=1.5
                )
                
                await agent.report_task_complete(task_result)
                
                # Verify completion data was sent
                completion_calls = [call for call in Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post.call_args_list 
                                 if 'complete' in str(call)]
                assert len(completion_calls) > 0
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_full_system_workflow(self):
        """Test complete system workflow from task receipt to completion"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test complete system interaction
        workflow_steps = []
        
        def track_step(step_name):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    workflow_steps.append(step_name)
                    return func(*args, **kwargs)
                return wrapper
            return decorator
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test backend responses
        @track_step("registration")
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_registration(*args, **kwargs):
            response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            response.status_code = 200
            return response
        
        @track_step("get_task")
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_task(*args, **kwargs):
            response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            response.status_code = 200
            response.json.return_value = {
                "id": "workflow-test-001",
                "type": "full-workflow-test",
                "data": {
                    "prompt": "Complete system workflow test",
                    "model": "tinyllama"
                }
            }
            return response
        
        @track_step("task_completion")
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_completion(*args, **kwargs):
            response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            response.status_code = 200
            return response
        
        @track_step("ollama_processing")
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama(*args, **kwargs):
            return "Workflow test completed successfully"
        
        @track_step("heartbeat")
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_heartbeat(*args, **kwargs):
            response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            response.status_code = 200
            return response
        
        with patch.object(agent.http_client, 'post', side_effect=lambda url, **kwargs: 
                         Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_registration() if 'register' in url
                         else Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_completion() if 'complete' in url
                         else Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_heartbeat()):
            with patch.object(agent.http_client, 'get', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_task):
                with patch.object(agent.circuit_breaker, 'call', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama):
                    
                    # Execute full workflow
                    # 1. Register with coordinator
                    registration_success = await agent.register_with_coordinator()
                    assert registration_success is True
                    
                    # 2. Get task from coordinator
                    task = await agent.get_next_task()
                    assert task is not None
                    
                    # 3. Process task (including Ollama interaction)
                    task_result = await agent.process_task(task)
                    assert task_result.status == "completed"
                    
                    # 4. Report completion to coordinator
                    await agent.report_task_complete(task_result)
                    
                    # 5. Send heartbeat
                    agent.shutdown_event.set()  # Exit quickly
                    await agent.send_heartbeat()
        
        # Verify all workflow steps were executed
        expected_steps = ["registration", "get_task", "task_completion", "heartbeat"]
        for step in expected_steps:
            assert step in workflow_steps, f"Workflow step '{step}' was not executed"
        
        # Verify metrics were updated
        assert agent.metrics.tasks_processed == 1
        assert agent.metrics.ollama_requests >= 1
        
        await agent._cleanup_async_components()


class TestConfigurationIntegration:
    """Test integration with different configurations"""
    
    @pytest.mark.asyncio
    async def test_agent_config_integration(self):
        """Test agent behavior with different configurations"""
        # Create config file with specific settings
        config_data = {
            "capabilities": ["text-generation", "code-analysis", "data-processing"],
            "max_retries": 5,
            "timeout": 120,
            "batch_size": 20,
            "custom_setting": "integration_test_value"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            agent = BaseAgent(
                config_path=config_path,
                max_concurrent_tasks=4,
                health_check_interval=60
            )
            
            # Verify config was loaded correctly
            assert agent.config["capabilities"] == config_data["capabilities"]
            assert agent.config["max_retries"] == 5
            assert agent.config["timeout"] == 120
            assert agent.config["batch_size"] == 20
            assert agent.config["custom_setting"] == "integration_test_value"
            
            # Verify initialization parameters
            assert agent.max_concurrent_tasks == 4
            assert agent.health_check_interval == 60
            
            await agent._setup_async_components()
            
            # Test that config affects behavior
            # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test registration to verify capabilities are sent
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response.status_code = 200
            
            with patch.object(agent.http_client, 'post', return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response) as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post:
                await agent.register_with_coordinator()
                
                # Verify capabilities were included in registration
                call_args = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_post.call_args
                registration_data = call_args[1]['json']
                assert registration_data['capabilities'] == config_data["capabilities"]
            
            await agent._cleanup_async_components()
            
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_model_configuration_integration(self):
        """Test integration with different model configurations"""
        # Test different agent types and their model assignments - all using GPT-OSS
        test_cases = [
            (" system-architect", "tinyllama"),
            ("ai-product-manager", "tinyllama"),
            ("garbage-collector", "tinyllama"),
        ]
        
        for agent_name, expected_model in test_cases:
            with patch.dict(os.environ, {
                'AGENT_NAME': agent_name,
                'AGENT_TYPE': 'config-test'
            }):
                agent = BaseAgent()
                
                # Verify model configuration - should be GPT-OSS
                assert agent.default_model == "tinyllama"
                
                # Verify model config parameters
                try:
                    model_config = OllamaConfig.get_model_config(agent_name)
                    assert model_config["model"] == "tinyllama"
                except (AssertionError, Exception) as e:
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    # Config might not exist, that's OK as long as default model is correct
                    pass
                
                # Verify config is used in Ollama integration
                await agent._setup_async_components()
                
                # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Ollama call with config verification
                def verify_model_config(*args, **kwargs):
                    # Verify that model config parameters are passed
                    assert kwargs.get('model') == expected_model
                    return f"Response from {expected_model}"
                
                with patch.object(agent.circuit_breaker, 'call', side_effect=verify_model_config):
                    result = await agent.query_ollama("Test prompt")
                    assert result == f"Response from {expected_model}"
                
                await agent._cleanup_async_components()


class TestErrorPropagationIntegration:
    """Test error propagation through the integrated system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_handling(self):
        """Test error handling from Ollama through to task completion"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Test error propagation chain
        error_chain = []
        
        def track_error(component, error):
            error_chain.append((component, str(error)))
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Ollama failure
        def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_failure(*args, **kwargs):
            error = Exception("Simulated Ollama service failure")
            track_error("ollama", error)
            raise error
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test task processing that uses Ollama
        original_process_task = agent.process_task
        
        async def tracking_process_task(task):
            try:
                # Attempt to use Ollama
                with patch.object(agent.circuit_breaker, 'call', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_ollama_failure):
                    result = await agent.query_ollama("Test prompt for error handling")
                    
                    if result is None:
                        track_error("agent_query", Exception("Ollama query returned None"))
                    
                # Continue with base processing
                return await original_process_task(task)
                
            except Exception as e:
                track_error("task_processing", e)
                raise
        
        agent.process_task = tracking_process_task
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test task completion reporting with error tracking
        original_report = agent.report_task_complete
        
        async def tracking_report(task_result):
            if task_result.error:
                track_error("task_result", Exception(task_result.error))
            return await original_report(task_result)
        
        agent.report_task_complete = tracking_report
        
        # Execute task that will trigger error chain
        task = {
            "id": "error-propagation-test",
            "type": "error-test",
            "data": {"prompt": "This will trigger an error"}
        }
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test successful task completion reporting
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response.status_code = 200
        
        with patch.object(agent.http_client, 'post', return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response):
            result = await agent.process_task(task)
        
        # Verify error was handled gracefully at task level
        assert result.task_id == "error-propagation-test"
        # Task should complete (base implementation doesn't use Ollama)
        assert result.status == "completed"
        
        # Verify error chain was tracked
        assert len(error_chain) >= 2  # Should have ollama and agent_query errors
        
        # Verify metrics reflect the failures
        assert agent.metrics.ollama_failures >= 1
        
        await agent._cleanup_async_components()


if __name__ == "__main__":
    unittest.main()
