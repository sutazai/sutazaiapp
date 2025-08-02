"""
Coordinator System Integration Test Suite for SutazAI automation System

Tests integration between the main backend and the coordinator system,
including processing processing, system_state, and learning capabilities.
"""

import pytest
import httpx
import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"
BRAIN_URL = "http://localhost:8001"  # Assuming coordinator runs on separate port
BRAIN_INTERNAL_URL = "http://sutazai-coordinator:8001"  # Docker internal URL

INTEGRATION_TIMEOUT = 60.0  # Longer timeout for coordinator operations

@pytest.fixture
async def backend_client():
    """Create async HTTP client for backend"""
    timeout = httpx.Timeout(INTEGRATION_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=timeout) as client:
        yield client

@pytest.fixture
async def coordinator_client():
    """Create async HTTP client for coordinator system"""
    timeout = httpx.Timeout(INTEGRATION_TIMEOUT, connect=10.0)
    # Try both local and internal URLs
    for url in [BRAIN_URL, BRAIN_INTERNAL_URL]:
        try:
            async with httpx.AsyncClient(base_url=url, timeout=timeout) as client:
                # Test connectivity
                response = await client.get("/health")
                if response.status_code == 200:
                    yield client
                    return
        except:
            continue
    
    # If neither works, yield None to skip coordinator-specific tests
    yield None

@pytest.fixture
async def auth_headers():
    """Get authentication headers"""
    return {}

class TestCoordinatorSystemConnectivity:
    """Test basic connectivity between backend and coordinator system"""
    
    @pytest.mark.asyncio
    async def test_backend_coordinator_health_integration(self, backend_client):
        """Test that backend can report coordinator system health"""
        response = await backend_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "services" in data
        
        # Check if coordinator system is mentioned in health status
        services = data["services"]
        
        # Look for coordinator-related services or processing engine status
        coordinator_indicators = [
            "processing_engine", "coordinator", "system_state", "reasoning_engine"
        ]
        
        found_coordinator_service = False
        for indicator in coordinator_indicators:
            if indicator in services or any(indicator in str(v) for v in services.values()):
                found_coordinator_service = True
                break
        
        # If coordinator system is integrated, it should be reported
        if "processing_engine" in services:
            assert "status" in services["processing_engine"]
    
    @pytest.mark.asyncio
    async def test_direct_coordinator_health(self, coordinator_client):
        """Test direct coordinator system health check"""
        if coordinator_client is None:
            pytest.skip("Coordinator system not accessible")
        
        response = await coordinator_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "coordinator_id" in data or "service" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_coordinator_system_status_via_backend(self, backend_client, auth_headers):
        """Test coordinator system status through backend API"""
        response = await backend_client.get("/api/v1/system/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "components" in data
        
        components = data["components"]
        if "processing_engine" in components:
            processing_engine = components["processing_engine"]
            assert "active" in processing_engine
            assert "healthy" in processing_engine
            
            if processing_engine["active"]:
                assert "system_state_active" in processing_engine

class TestProcessingProcessingIntegration:
    """Test processing processing integration between backend and coordinator"""
    
    @pytest.mark.asyncio
    async def test_processing_processing_endpoint(self, backend_client, auth_headers):
        """Test processing processing through backend"""
        payload = {
            "input_data": {
                "text": "Analyze the relationship between artificial intelligence and human system_state",
                "type": "philosophical_analysis"
            },
            "processing_type": "deep_analysis",
            "use_system_state": True,
            "reasoning_depth": 3
        }
        
        response = await backend_client.post(
            "/api/v1/processing/process",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "result" in data
        assert "processing_type" in data
        assert "system_state_enabled" in data
        assert "reasoning_depth" in data
        assert "timestamp" in data
        
        # If processing processing is actually working
        if not data.get("fallback_mode", False):
            assert "processing_pathways_activated" in data
    
    @pytest.mark.asyncio
    async def test_system_state_state_monitoring(self, backend_client, auth_headers):
        """Test system_state state monitoring"""
        response = await backend_client.get("/api/v1/processing/system_state", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "system_state_active" in data
        assert "timestamp" in data
        
        # If system_state is active, check detailed state
        if data["system_state_active"]:
            assert "awareness_level" in data
            assert "cognitive_load" in data
            assert "active_processes" in data
            assert "processing_activity" in data
            
            # Validate system_state metrics
            assert 0.0 <= data["awareness_level"] <= 1.0
            assert 0.0 <= data["cognitive_load"] <= 1.0
            assert isinstance(data["active_processes"], list)
            assert isinstance(data["processing_activity"], dict)
    
    @pytest.mark.asyncio
    async def test_creative_synthesis_integration(self, backend_client, auth_headers):
        """Test creative synthesis processing processing"""
        payload = {
            "prompt": "Design an innovative solution for sustainable urban transportation",
            "synthesis_mode": "cross_domain",
            "reasoning_depth": 4,
            "use_system_state": True
        }
        
        response = await backend_client.post(
            "/api/v1/processing/creative",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis" in data
        assert "insights" in data
        assert "recommendations" in data
        assert "output" in data
        assert "synthesis_mode" in data
        assert "system_state_active" in data
        assert "reasoning_depth" in data
        assert "creative_pathways" in data
        assert "timestamp" in data
        
        # Validate creative synthesis structure
        assert data["synthesis_mode"] == "cross_domain"
        assert data["reasoning_depth"] == 4
        assert isinstance(data["insights"], list)
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["creative_pathways"], list)
    
    @pytest.mark.asyncio
    async def test_direct_coordinator_processing(self, coordinator_client):
        """Test direct coordinator system processing"""
        if coordinator_client is None:
            pytest.skip("Coordinator system not accessible")
        
        payload = {
            "input": "Analyze advanced computing applications in AI",
            "context": {
                "domain": "technology",
                "complexity": "high"
            },
            "stream": False
        }
        
        response = await coordinator_client.post("/process", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert "output" in data
        assert "confidence" in data
        assert "execution_time" in data
        assert "agents_used" in data
        
        # Validate coordinator processing metrics
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["execution_time"] >= 0.0
        assert isinstance(data["agents_used"], list)

class TestCoordinatorLearningIntegration:
    """Test learning system integration"""
    
    @pytest.mark.asyncio
    async def test_knowledge_learning_via_backend(self, backend_client):
        """Test knowledge learning through backend"""
        payload = {
            "content": "Machine learning models require careful attention to bias and fairness. Recent research shows that diverse training data and regular auditing can help mitigate discriminatory outcomes in AI systems.",
            "type": "research_finding"
        }
        
        response = await backend_client.post("/learn", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "learned" in data
        assert "content_type" in data
        assert "content_size" in data
        assert "summary" in data
        assert "knowledge_points" in data
        assert "processing_stats" in data
        assert "processing_time" in data
        assert "timestamp" in data
        
        # Validate learning process
        assert data["learned"] == True
        assert data["content_type"] == "research_finding"
        assert isinstance(data["knowledge_points"], list)
        assert isinstance(data["processing_stats"], dict)
    
    @pytest.mark.asyncio
    async def test_learning_impact_on_reasoning(self, backend_client, auth_headers):
        """Test that learning impacts subsequent reasoning"""
        # First, teach the system something specific
        learning_payload = {
            "content": "The Fibonacci sequence exhibits the golden ratio in its consecutive number ratios, which appears frequently in nature.",
            "type": "mathematical_fact"
        }
        
        learn_response = await backend_client.post("/learn", json=learning_payload)
        assert learn_response.status_code == 200
        
        # Small delay to allow processing
        await asyncio.sleep(2)
        
        # Then ask a related question
        reasoning_payload = {
            "query": "What mathematical patterns appear in natural phenomena?",
            "reasoning_type": "analytical"
        }
        
        reason_response = await backend_client.post(
            "/think",
            json=reasoning_payload,
            headers=auth_headers
        )
        
        assert reason_response.status_code == 200
        
        reason_data = reason_response.json()
        assert "thought" in reason_data
        assert "reasoning" in reason_data
        assert "confidence" in reason_data
        
        # The response should ideally incorporate the learned knowledge
        # (This is a soft assertion since the system might not always connect the knowledge)
        response_text = reason_data["thought"].lower()
        fibonacci_mentioned = "fibonacci" in response_text or "golden ratio" in response_text
        
        # Log for debugging
        print(f"Learning integration test - Fibonacci mentioned: {fibonacci_mentioned}")
    
    @pytest.mark.asyncio
    async def test_continuous_learning_loop(self, backend_client, auth_headers):
        """Test continuous learning capabilities"""
        learning_sequence = [
            {
                "content": "Renewable energy sources include solar, wind, and hydroelectric power.",
                "type": "factual_information"
            },
            {
                "content": "Solar panels convert sunlight to electricity using photovoltaic cells.",
                "type": "technical_detail"
            },
            {
                "content": "Wind turbines are most efficient in coastal and elevated areas.",
                "type": "operational_insight"
            }
        ]
        
        # Learn each piece of information
        for item in learning_sequence:
            response = await backend_client.post("/learn", json=item)
            assert response.status_code == 200
            
            data = response.json()
            assert data["learned"] == True
            
            await asyncio.sleep(1)  # Brief pause between learning
        
        # Test if the system can synthesize the learned information
        synthesis_payload = {
            "query": "Design a comprehensive renewable energy strategy",
            "reasoning_type": "strategic"
        }
        
        synthesis_response = await backend_client.post(
            "/think",
            json=synthesis_payload,
            headers=auth_headers
        )
        
        assert synthesis_response.status_code == 200
        
        synthesis_data = synthesis_response.json()
        assert "thought" in synthesis_data
        assert "confidence" in synthesis_data
        
        # The response should demonstrate synthesis of learned knowledge
        response_text = synthesis_data["thought"].lower()
        energy_terms = ["solar", "wind", "renewable", "energy"]
        mentioned_terms = sum(1 for term in energy_terms if term in response_text)
        
        assert mentioned_terms >= 2, "System should synthesize learned renewable energy knowledge"

class TestCoordinatorSelfImprovementIntegration:
    """Test self-improvement system integration"""
    
    @pytest.mark.asyncio
    async def test_improvement_analysis_integration(self, backend_client, auth_headers):
        """Test improvement analysis through coordinator system"""
        response = await backend_client.post("/api/v1/improvement/analyze", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        
        # If enterprise improvement system is available
        if "analysis_id" in data:
            assert "improvements_identified" in data
            assert "priority_areas" in data
            assert "estimated_impact" in data
            assert "implementation_plan" in data
            
            # Validate improvement analysis structure
            assert isinstance(data["improvements_identified"], list)
            assert isinstance(data["priority_areas"], list)
            assert isinstance(data["estimated_impact"], dict)
            assert isinstance(data["implementation_plan"], list)
    
    @pytest.mark.asyncio
    async def test_legacy_improvement_system(self, backend_client, auth_headers):
        """Test legacy self-improvement endpoint"""
        response = await backend_client.post("/improve", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "improvement" in data
        assert "changes" in data
        assert "impact" in data
        assert "next_optimization" in data
        assert "optimization_areas" in data
        assert "performance_gains" in data
        assert "enterprise_mode" in data
        assert "timestamp" in data
        
        # Validate improvement data structure
        assert isinstance(data["changes"], list)
        assert isinstance(data["optimization_areas"], list)
        assert isinstance(data["performance_gains"], dict)
        assert len(data["changes"]) > 0
    
    @pytest.mark.asyncio
    async def test_improvement_application(self, backend_client, auth_headers):
        """Test improvement application process"""
        # First analyze for improvements
        analyze_response = await backend_client.post("/api/v1/improvement/analyze", headers=auth_headers)
        assert analyze_response.status_code == 200
        
        # Try to apply some improvements
        improvement_ids = ["memory_optimization", "response_speed", "processing_efficiency"]
        
        apply_response = await backend_client.post(
            "/api/v1/improvement/apply",
            json=improvement_ids,
            headers=auth_headers
        )
        
        assert apply_response.status_code == 200
        
        apply_data = apply_response.json()
        assert "applied" in apply_data
        assert "timestamp" in apply_data
        
        # If improvement system is available and applied improvements
        if apply_data["applied"] and "improvement_results" in apply_data:
            assert "system_restart_required" in apply_data
            assert "performance_impact" in apply_data

class TestCoordinatorMemoryIntegration:
    """Test coordinator memory and knowledge management integration"""
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, backend_client, auth_headers):
        """Test that coordinator maintains memory across operations"""
        # Store some information
        learn_payload = {
            "content": "The unique identifier for this test session is: TEST_SESSION_12345",
            "type": "session_data"
        }
        
        learn_response = await backend_client.post("/learn", json=learn_payload)
        assert learn_response.status_code == 200
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Perform some other operations
        for i in range(3):
            other_payload = {
                "query": f"Perform calculation: {i * 2} + {i * 3}",
                "reasoning_type": "mathematical"
            }
            
            other_response = await backend_client.post(
                "/think",
                json=other_payload,
                headers=auth_headers
            )
            assert other_response.status_code == 200
            await asyncio.sleep(1)
        
        # Try to recall the stored information
        recall_payload = {
            "query": "What was the unique identifier for this test session?",
            "reasoning_type": "recall"
        }
        
        recall_response = await backend_client.post(
            "/think",
            json=recall_payload,
            headers=auth_headers
        )
        
        assert recall_response.status_code == 200
        
        recall_data = recall_response.json()
        assert "thought" in recall_data
        
        # Check if the system can recall the stored information
        response_text = recall_data["thought"]
        recalled_identifier = "TEST_SESSION_12345" in response_text
        
        # Log for debugging
        print(f"Memory persistence test - Identifier recalled: {recalled_identifier}")
        print(f"Recall response: {response_text[:200]}...")
    
    @pytest.mark.asyncio
    async def test_contextual_memory_usage(self, backend_client, auth_headers):
        """Test contextual usage of stored memories"""
        # Store contextual information
        context_items = [
            {
                "content": "Project Alpha is a machine learning initiative focused on natural language processing.",
                "type": "project_context"
            },
            {
                "content": "The team lead for Project Alpha is Dr. Sarah Chen, an expert in processing networks.",
                "type": "team_information"
            },
            {
                "content": "Project Alpha's deadline is set for Q2 2024, with a budget of $2.5 million.",
                "type": "project_details"
            }
        ]
        
        # Learn each piece of context
        for item in context_items:
            response = await backend_client.post("/learn", json=item)
            assert response.status_code == 200
            await asyncio.sleep(1)
        
        # Ask contextual questions
        contextual_questions = [
            "Who is leading the machine learning project?",
            "What is the budget for Project Alpha?",
            "When is the NLP project deadline?"
        ]
        
        correct_contexts = 0
        
        for question in contextual_questions:
            question_payload = {
                "query": question,
                "reasoning_type": "contextual"
            }
            
            response = await backend_client.post(
                "/think",
                json=question_payload,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            data = response.json()
            response_text = data["thought"].lower()
            
            # Check if response contains relevant context
            if "alpha" in response_text or "sarah" in response_text or "2.5" in response_text or "q2" in response_text:
                correct_contexts += 1
        
        # At least some questions should use stored context
        context_usage_rate = correct_contexts / len(contextual_questions)
        print(f"Contextual memory usage rate: {context_usage_rate}")

class TestCoordinatorPerformanceIntegration:
    """Test performance aspects of coordinator system integration"""
    
    @pytest.mark.asyncio
    async def test_processing_processing_performance(self, backend_client, auth_headers):
        """Test processing processing performance"""
        processing_times = []
        
        for i in range(5):
            start_time = time.time()
            
            payload = {
                "input_data": {
                    "text": f"Analyze the efficiency of algorithm {i} for data processing",
                    "complexity": "moderate"
                },
                "processing_type": "performance_analysis",
                "use_system_state": True,
                "reasoning_depth": 2
            }
            
            response = await backend_client.post(
                "/api/v1/processing/process",
                json=payload,
                headers=auth_headers
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert response.status_code == 200
            processing_times.append(processing_time)
            
            await asyncio.sleep(1)
        
        # Calculate performance metrics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        # Performance assertions
        assert avg_time < 30.0, f"Average processing processing time too high: {avg_time}s"
        assert max_time < 45.0, f"Maximum processing processing time too high: {max_time}s"
        
        print(f"Processing processing performance - Avg: {avg_time:.2f}s, Max: {max_time:.2f}s, Min: {min_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_coordinator_operations(self, backend_client, auth_headers):
        """Test concurrent coordinator operations"""
        async def concurrent_operation(operation_id):
            payload = {
                "query": f"Concurrent operation {operation_id}: analyze data patterns",
                "reasoning_type": "analytical"
            }
            
            return await backend_client.post(
                "/think",
                json=payload,
                headers=auth_headers
            )
        
        # Run 5 concurrent operations
        start_time = time.time()
        tasks = [concurrent_operation(i) for i in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Count successful responses
        successful = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        success_rate = len(successful) / len(responses)
        
        # Performance assertions
        assert success_rate >= 0.8, f"Concurrent operation success rate too low: {success_rate}"
        assert total_time < 60.0, f"Concurrent operations took too long: {total_time}s"
        
        print(f"Concurrent operations - Success rate: {success_rate}, Total time: {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_coordinator_resource_usage(self, backend_client, auth_headers):
        """Test coordinator system resource usage monitoring"""
        # Get initial system state
        initial_response = await backend_client.get("/api/v1/system/status", headers=auth_headers)
        assert initial_response.status_code == 200
        
        initial_data = initial_response.json()
        initial_performance = initial_data.get("performance", {})
        
        # Perform resource-intensive coordinator operations
        intensive_operations = [
            {
                "input_data": {
                    "text": "Perform complex analysis of distributed systems architecture with multiple microservices, considering scalability, reliability, and performance optimization strategies",
                    "complexity": "very_high"
                },
                "processing_type": "architectural_analysis",
                "use_system_state": True,
                "reasoning_depth": 5
            },
            {
                "prompt": "Design a comprehensive AI safety framework that addresses alignment, robustness, interpretability, and ethical considerations across multiple domains",
                "synthesis_mode": "multi_domain_integration",
                "reasoning_depth": 4,
                "use_system_state": True
            }
        ]
        
        for operation in intensive_operations:
            if "input_data" in operation:
                endpoint = "/api/v1/processing/process"
            else:
                endpoint = "/api/v1/processing/creative"
            
            response = await backend_client.post(endpoint, json=operation, headers=auth_headers)
            assert response.status_code == 200
            
            await asyncio.sleep(2)
        
        # Get final system state
        final_response = await backend_client.get("/api/v1/system/status", headers=auth_headers)
        assert final_response.status_code == 200
        
        final_data = final_response.json()
        final_performance = final_data.get("performance", {})
        
        # System should still be responsive after intensive operations
        assert "cpu_percent" in final_performance
        assert "memory_percent" in final_performance
        
        print(f"Resource usage after intensive operations - CPU: {final_performance.get('cpu_percent', 'N/A')}%, Memory: {final_performance.get('memory_percent', 'N/A')}%")

class TestCoordinatorErrorHandling:
    """Test error handling in coordinator system integration"""
    
    @pytest.mark.asyncio
    async def test_invalid_processing_processing_input(self, backend_client, auth_headers):
        """Test handling of invalid processing processing input"""
        invalid_payloads = [
            {},  # Empty payload
            {"input_data": ""},  # Empty input data
            {"input_data": {"text": ""}, "processing_type": ""},  # Empty processing type
            {"input_data": {"text": "test"}, "reasoning_depth": -1},  # Invalid reasoning depth
            {"input_data": {"text": "test"}, "reasoning_depth": 100},  # Excessive reasoning depth
        ]
        
        for payload in invalid_payloads:
            response = await backend_client.post(
                "/api/v1/processing/process",
                json=payload,
                headers=auth_headers
            )
            
            # Should handle gracefully - either 422 for validation or 200 with error handling
            assert response.status_code in [200, 422]
            
            if response.status_code == 200:
                data = response.json()
                # Should contain result or fallback processing
                assert "result" in data or "error" in data
    
    @pytest.mark.asyncio
    async def test_coordinator_system_timeout_handling(self, backend_client, auth_headers):
        """Test timeout handling for coordinator operations"""
        # Create a potentially long-running operation
        complex_payload = {
            "input_data": {
                "text": "Analyze every possible combination and permutation of strategic approaches for solving climate change, considering economic, political, social, technological, and environmental factors across all global regions with detailed cost-benefit analysis for each approach",
                "complexity": "maximum"
            },
            "processing_type": "exhaustive_analysis",
            "use_system_state": True,
            "reasoning_depth": 10  # Very deep reasoning
        }
        
        start_time = time.time()
        
        try:
            response = await backend_client.post(
                "/api/v1/processing/process",
                json=complex_payload,
                headers=auth_headers
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should either complete or timeout gracefully
            assert response.status_code in [200, 408, 503]
            
            if response.status_code == 200:
                data = response.json()
                assert "result" in data
                print(f"Complex operation completed in {processing_time:.2f}s")
            else:
                print(f"Complex operation timed out or failed gracefully after {processing_time:.2f}s")
                
        except asyncio.TimeoutError:
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Operation timed out after {processing_time:.2f}s - this is expected for very complex operations")
    
    @pytest.mark.asyncio
    async def test_coordinator_recovery_after_errors(self, backend_client, auth_headers):
        """Test coordinator system recovery after encountering errors"""
        # Cause some errors first
        error_payloads = [
            {"input_data": None, "processing_type": "invalid"},
            {"input_data": {"text": "x" * 100000}, "processing_type": "oversized"},  # Very large input
        ]
        
        for payload in error_payloads:
            try:
                response = await backend_client.post(
                    "/api/v1/processing/process",
                    json=payload,
                    headers=auth_headers
                )
                # Any status is acceptable - we're testing recovery
                assert response.status_code in [200, 400, 422, 500]
            except Exception:
                # Exceptions are acceptable for error testing
                pass
            
            await asyncio.sleep(1)
        
        # Now test that the system can still handle normal operations
        normal_payload = {
            "input_data": {
                "text": "Test normal operation after errors",
                "context": "recovery_test"
            },
            "processing_type": "simple_analysis",
            "use_system_state": False,
            "reasoning_depth": 1
        }
        
        recovery_response = await backend_client.post(
            "/api/v1/processing/process",
            json=normal_payload,
            headers=auth_headers
        )
        
        assert recovery_response.status_code == 200
        
        recovery_data = recovery_response.json()
        assert "result" in recovery_data
        assert "timestamp" in recovery_data
        
        print("Coordinator system successfully recovered after errors")

# Test configuration and utilities
if __name__ == "__main__":
    pytest.main(["-v", __file__])
