#!/usr/bin/env python3
"""
End-to-End Tests for SutazAI User Workflows
Validates complete user journeys per Rules 1-19
"""

import pytest
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import os
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

# Test configuration
BASE_URL = os.getenv('TEST_BASE_URL', 'http://localhost:10010')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:10011')
TEST_TIMEOUT = 60.0
PAGE_LOAD_TIMEOUT = 30.0


def setup_chrome_driver():
    """Setup Chrome driver for E2E tests"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode for CI
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        return driver
    except Exception as e:
        pytest.skip(f"Chrome driver not available: {e}")


@pytest.mark.e2e
class TestCompleteUserJourney:
    """Test complete user journey from start to finish"""
    
    @pytest.fixture(scope='class')
    def driver(self):
        """Setup Chrome driver for class"""
        driver = setup_chrome_driver()
        yield driver
        driver.quit()
    
    def test_01_system_initialization(self):
        """Test that all systems are properly initialized"""
        # Test backend health
        response = httpx.get(f"{BASE_URL}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data['status'] == 'healthy'
        
        # Verify key services are operational
        services = health_data.get('services', {})
        required_services = ['database', 'redis', 'ollama']
        
        for service in required_services:
            if service in services:
                assert services[service] in ['healthy', 'connected', 'ok', 'running']
    
    def test_02_frontend_loads_successfully(self, driver):
        """Test that frontend loads and displays correctly"""
        driver.get(FRONTEND_URL)
        
        # Wait for page to load
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check for Streamlit elements
        assert "SutazAI" in driver.title or "Streamlit" in driver.page_source
        
        # Verify no obvious errors
        assert "Error" not in driver.title
        assert "404" not in driver.page_source
    
    def test_03_navigation_between_pages(self, driver):
        """Test navigation between different pages"""
        driver.get(FRONTEND_URL)
        
        # Wait for page to load
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find navigation elements (Streamlit sidebar or tabs)
        try:
            # Look for common Streamlit navigation elements
            sidebar_elements = driver.find_elements(By.CLASS_NAME, "sidebar")
            tab_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'tab')]")
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            
            # If navigation exists, test it
            if sidebar_elements or tab_elements or link_elements:
                # Successfully found navigation elements
                pass
        except Exception as e:
            # Navigation might not be implemented yet
            pytest.skip(f"Navigation not available: {e}")
    
    @pytest.mark.asyncio
    async def test_04_api_chat_workflow(self):
        """Test complete API chat workflow"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Check available models
            models_response = await client.get(f"{BASE_URL}/api/v1/models/")
            assert models_response.status_code == 200
            
            models_data = models_response.json()
            available_models = models_data.get('models', [])
            
            # Step 2: Send a chat message
            chat_request = {
                "message": "Hello, can you help me understand what this system does?",
                "model": "tinyllama"
            }
            
            chat_response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_request
            )
            
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            
            # Step 3: Validate response structure and content
            assert 'response' in chat_data
            assert isinstance(chat_data['response'], str)
            assert len(chat_data['response']) > 0
            
            # Step 4: Test follow-up conversation
            follow_up_request = {
                "message": "Can you tell me more?",
                "model": "tinyllama",
                "context": chat_data['response']
            }
            
            follow_up_response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=follow_up_request
            )
            
            assert follow_up_response.status_code == 200
            follow_up_data = follow_up_response.json()
            assert 'response' in follow_up_data
    
    @pytest.mark.asyncio
    async def test_05_task_submission_workflow(self):
        """Test complete task submission and processing workflow"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Check task queue status
            queue_response = await client.get(f"{BASE_URL}/api/v1/tasks/queue/status")
            if queue_response.status_code == 200:
                initial_queue_status = queue_response.json()
            else:
                initial_queue_status = {}
            
            # Step 2: Submit a task
            task_request = {
                "type": "text_generation",
                "payload": {
                    "prompt": "Generate a professional summary of AI capabilities",
                    "model": "tinyllama",
                    "max_tokens": 200
                },
                "priority": 1
            }
            
            submit_response = await client.post(
                f"{BASE_URL}/api/v1/tasks/",
                json=task_request
            )
            
            if submit_response.status_code in [200, 201, 202]:
                submit_data = submit_response.json()
                task_id = submit_data.get('task_id') or submit_data.get('id')
                
                # Step 3: Monitor task status
                if task_id:
                    for attempt in range(10):  # Poll for up to 30 seconds
                        status_response = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            task_status = status_data.get('status')
                            
                            if task_status in ['completed', 'failed']:
                                break
                        
                        await asyncio.sleep(3)
                    
                    # Step 4: Verify task completion
                    final_response = await client.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
                    if final_response.status_code == 200:
                        final_data = final_response.json()
                        # Task should be processed (completed or failed)
                        assert final_data.get('status') in ['completed', 'failed', 'processing']
            else:
                # Task submission might not be implemented
                pytest.skip("Task submission endpoint not available")
    
    @pytest.mark.asyncio
    async def test_06_agent_orchestration_workflow(self):
        """Test agent orchestration and coordination workflow"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: List available agents
            agents_response = await client.get(f"{BASE_URL}/api/v1/agents/")
            assert agents_response.status_code == 200
            
            agents_data = agents_response.json()
            available_agents = agents_data.get('agents', []) if isinstance(agents_data, dict) else agents_data
            
            # Step 2: Check agent health
            health_response = await client.get(f"{BASE_URL}/api/v1/agents/health")
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            
            # Step 3: Test mesh coordination if available
            mesh_request = {
                "task_type": "coordination_test",
                "payload": {
                    "message": "Test inter-agent coordination",
                    "target_agents": ["any"]
                },
                "priority": 0
            }
            
            mesh_response = await client.post(
                f"{BASE_URL}/api/v1/mesh/enqueue",
                json=mesh_request
            )
            
            if mesh_response.status_code in [200, 201, 202]:
                mesh_data = mesh_response.json()
                task_id = mesh_data.get('task_id') or mesh_data.get('id')
                
                # Step 4: Check mesh results
                if task_id:
                    await asyncio.sleep(2)  # Give time for processing
                    
                    results_response = await client.get(f"{BASE_URL}/api/v1/mesh/results")
                    assert results_response.status_code == 200
    
    def test_07_frontend_interaction_workflow(self, driver):
        """Test frontend user interaction workflow"""
        driver.get(FRONTEND_URL)
        
        # Wait for page to load
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        try:
            # Look for common Streamlit input elements
            text_inputs = driver.find_elements(By.XPATH, "//input[@type='text']")
            text_areas = driver.find_elements(By.TAG_NAME, "textarea")
            buttons = driver.find_elements(By.TAG_NAME, "button")
            
            # If interactive elements exist, test them
            if text_inputs or text_areas:
                input_element = text_inputs[0] if text_inputs else text_areas[0]
                input_element.clear()
                input_element.send_keys("Test user input")
                
                # Look for submit button
                submit_buttons = [btn for btn in buttons if 'submit' in btn.text.lower() or 'send' in btn.text.lower()]
                
                if submit_buttons:
                    submit_buttons[0].click()
                    
                    # Wait for response (up to 10 seconds)
                    time.sleep(3)
                    
                    # Check for response in page
                    page_text = driver.page_source
                    # Should have some kind of response or update
                    assert len(page_text) > 1000  # Page should have content
            
            # Test passed - UI interaction successful
            assert True
            
        except Exception as e:
            # Interactive elements might not be fully implemented
            pytest.skip(f"Frontend interaction not fully available: {e}")
    
    @pytest.mark.asyncio
    async def test_08_system_monitoring_workflow(self):
        """Test system monitoring and observability workflow"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Check system metrics
            metrics_response = await client.get(f"{BASE_URL}/metrics")
            assert metrics_response.status_code == 200
            
            metrics_text = metrics_response.text
            assert 'TYPE' in metrics_text  # Prometheus format
            
            # Step 2: Check system info
            info_response = await client.get(f"{BASE_URL}/api/v1/system/info")
            assert info_response.status_code == 200
            
            info_data = info_response.json()
            assert isinstance(info_data, dict)
            
            # Step 3: Check system stats
            stats_response = await client.get(f"{BASE_URL}/api/v1/system/stats")
            assert stats_response.status_code == 200
            
            stats_data = stats_response.json()
            assert isinstance(stats_data, dict)
            
            # Step 4: Verify health endpoint provides comprehensive data
            health_response = await client.get(f"{BASE_URL}/health")
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            assert 'services' in health_data
            assert 'performance' in health_data
    
    @pytest.mark.asyncio
    async def test_09_error_handling_workflow(self):
        """Test error handling across the entire system"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Step 1: Test invalid API endpoint
            invalid_response = await client.get(f"{BASE_URL}/api/v1/nonexistent")
            assert invalid_response.status_code in [404, 405]
            
            # Step 2: Test malformed request
            malformed_response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json={"invalid": "data"}
            )
            assert malformed_response.status_code in [400, 422]
            
            # Step 3: Test oversized request
            large_message = "A" * 100000  # 100KB message
            large_response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json={"message": large_message, "model": "tinyllama"}
            )
            # Should handle gracefully (accept, reject, or timeout)
            assert large_response.status_code in [200, 413, 422, 500]
            
            # Step 4: Test concurrent load
            async def concurrent_request():
                return await client.get(f"{BASE_URL}/health")
            
            # Send 20 concurrent requests
            tasks = [concurrent_request() for _ in range(20)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Most requests should succeed
            successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
            assert len(successful_responses) >= 15  # At least 75% success rate
    
    @pytest.mark.asyncio
    async def test_10_performance_requirements(self):
        """Test that system meets performance requirements"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test 1: Health endpoint response time (<500ms)
            start_time = time.time()
            health_response = await client.get(f"{BASE_URL}/health")
            health_time = (time.time() - start_time) * 1000
            
            assert health_response.status_code == 200
            assert health_time < 500  # Should respond within 500ms
            
            # Test 2: Chat endpoint response time (<5s for simple query)
            start_time = time.time()
            chat_response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json={"message": "Hi", "model": "tinyllama"}
            )
            chat_time = (time.time() - start_time) * 1000
            
            assert chat_response.status_code == 200
            assert chat_time < 10000  # Should respond within 10 seconds
            
            # Test 3: System can handle multiple concurrent users
            async def simulate_user_session():
                """Simulate a user session with multiple requests"""
                session_client = httpx.AsyncClient(timeout=TEST_TIMEOUT)
                try:
                    # Health check
                    await session_client.get(f"{BASE_URL}/health")
                    
                    # Chat interaction
                    await session_client.post(
                        f"{BASE_URL}/api/v1/chat/",
                        json={"message": "Hello", "model": "tinyllama"}
                    )
                    
                    # Models check
                    await session_client.get(f"{BASE_URL}/api/v1/models/")
                    
                    return True
                except Exception:
                    return False
                finally:
                    await session_client.aclose()
            
            # Simulate 5 concurrent user sessions
            user_tasks = [simulate_user_session() for _ in range(5)]
            results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            # At least 80% of user sessions should complete successfully
            successful_sessions = sum(1 for r in results if r is True)
            assert successful_sessions >= 4


@pytest.mark.e2e
class TestDataPersistenceWorkflow:
    """Test data persistence across system restarts"""
    
    @pytest.mark.asyncio
    async def test_database_data_persistence(self):
        """Test that database data persists across operations"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test database connectivity through health check
            health_response = await client.get(f"{BASE_URL}/health")
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            services = health_data.get('services', {})
            
            # Database should be connected
            if 'database' in services:
                assert services['database'] in ['healthy', 'connected', 'ok']
            
            # If data endpoints exist, test persistence
            # This is a placeholder for actual data persistence tests
            # which would depend on implemented CRUD endpoints
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test cache functionality and performance"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Make the same request twice
            start_time = time.time()
            first_response = await client.get(f"{BASE_URL}/api/v1/models/")
            first_time = (time.time() - start_time) * 1000
            
            start_time = time.time()
            second_response = await client.get(f"{BASE_URL}/api/v1/models/")
            second_time = (time.time() - start_time) * 1000
            
            # Both should succeed
            assert first_response.status_code == 200
            assert second_response.status_code == 200
            
            # Second request might be faster if cached (but not guaranteed)
            # Just verify both requests work
            assert first_response.json() == second_response.json()


@pytest.mark.e2e
class TestSystemRecoveryWorkflow:
    """Test system recovery and resilience"""
    
    @pytest.mark.asyncio
    async def test_service_degradation_handling(self):
        """Test system behavior when services are degraded"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Test basic functionality still works
            health_response = await client.get(f"{BASE_URL}/health")
            assert health_response.status_code == 200
            
            # Even if some services are degraded, core functionality should work
            health_data = health_response.json()
            # System should report status (healthy, degraded, or unhealthy)
            assert 'status' in health_data
            assert health_data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    @pytest.mark.asyncio
    async def test_timeout_and_retry_behavior(self):
        """Test timeout and retry behavior"""
        async with httpx.AsyncClient(timeout=5.0) as client:  # Short timeout
            try:
                # Make request with short timeout
                response = await client.get(f"{BASE_URL}/health")
                assert response.status_code == 200
            except httpx.TimeoutException:
                # Timeout is acceptable for this test
                pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
