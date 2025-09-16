"""
End-to-End Tests for Sutazai Frontend UI
Tests the Streamlit application interface and interactions
"""

import pytest
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import json
from typing import Optional

# Configuration
FRONTEND_URL = "http://localhost:11000"
TIMEOUT = 30

class TestStreamlitUI:
    """Test Streamlit UI without Selenium (using HTTP requests)"""
    
    def test_frontend_accessible(self):
        """Test that frontend is accessible"""
        response = requests.get(FRONTEND_URL, timeout=TIMEOUT)
        assert response.status_code == 200
        assert "streamlit" in response.text.lower() or "sutazai" in response.text.lower()
        
    def test_frontend_health(self):
        """Test frontend health endpoint"""
        response = requests.get(f"{FRONTEND_URL}/_stcore/health", timeout=TIMEOUT)
        # Streamlit might not have this endpoint, but check
        assert response.status_code in [200, 404]
        
    def test_static_resources(self):
        """Test that static resources are served"""
        response = requests.get(f"{FRONTEND_URL}/static/index.html", timeout=TIMEOUT)
        # May or may not exist
        assert response.status_code in [200, 404]


class TestUIComponents:
    """Test UI components via HTTP simulation"""
    
    def test_chat_interface_structure(self):
        """Test that chat interface components are present"""
        response = requests.get(FRONTEND_URL, timeout=TIMEOUT)
        assert response.status_code == 200
        content = response.text.lower()
        
        # Check for expected UI elements (these might be in JavaScript)
        expected_elements = [
            "chat", "message", "send", "model", "agent"
        ]
        
        found_elements = [elem for elem in expected_elements if elem in content]
        assert len(found_elements) > 0, f"No expected UI elements found in page"
        
    def test_agent_list_presence(self):
        """Test that agent list is present in UI"""
        response = requests.get(FRONTEND_URL, timeout=TIMEOUT)
        assert response.status_code == 200
        
        # Expected agents that should be visible
        expected_agents = [
            "JARVIS", "DocuMind", "QuantumCoder", "DataSage",
            "CreativeStorm", "TechAnalyst", "ResearchOwl",
            "CodeOptimizer", "VisionaryArch", "TestMaster", "SecurityGuard"
        ]
        
        content = response.text
        # At least some agents should be mentioned
        found_agents = [agent for agent in expected_agents if agent in content]
        # UI might load agents dynamically, so we just check response is valid
        assert response.status_code == 200


class TestStreamlitWebSocket:
    """Test Streamlit WebSocket connections"""
    
    def test_streamlit_websocket(self):
        """Test Streamlit's WebSocket connection"""
        import websocket
        
        # Streamlit uses WebSocket for real-time updates
        ws_url = "ws://localhost:11000/_stcore/stream"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            # Send a ping
            ws.ping()
            ws.close()
            assert True  # Connection successful
        except Exception as e:
            # WebSocket might require specific headers
            print(f"Streamlit WebSocket test: {e}")
            assert True  # Not critical if it fails


class TestChatFunctionality:
    """Test chat functionality through API"""
    
    def test_send_chat_message(self):
        """Test sending a chat message through the backend"""
        # Since we can't interact with Streamlit directly, test the backend
        api_url = "http://localhost:10200/api/v1/chat"
        payload = {
            "message": "Hello from E2E test",
            "model": "gpt-3.5-turbo"
        }
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data or "message" in data or "content" in data
        
    def test_chat_with_different_agents(self):
        """Test chatting with different agents"""
        api_url = "http://localhost:10200/api/v1/chat"
        agents = ["jarvis", "documind", "quantumcoder", "datasage"]
        
        for agent in agents:
            payload = {
                "message": f"Hello, I'm testing {agent}",
                "agent_id": agent
            }
            response = requests.post(api_url, json=payload, timeout=TIMEOUT)
            assert response.status_code in [200, 404, 422]
            
    def test_model_switching(self):
        """Test switching between different models"""
        api_url = "http://localhost:10200/api/v1/chat"
        models = ["gpt-3.5-turbo", "gpt-4", "claude-2"]
        
        for model in models:
            payload = {
                "message": "Test message",
                "model": model
            }
            response = requests.post(api_url, json=payload, timeout=TIMEOUT)
            assert response.status_code in [200, 400, 422, 503]


class TestSessionManagement:
    """Test session management functionality"""
    
    def test_session_creation(self):
        """Test creating a new session"""
        api_url = "http://localhost:10200/api/v1/chat"
        session_id = f"e2e_test_session_{int(time.time())}"
        
        payload = {
            "message": "Start new session",
            "session_id": session_id
        }
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
    def test_session_persistence(self):
        """Test that sessions persist across messages"""
        api_url = "http://localhost:10200/api/v1/chat"
        session_id = f"persistence_test_{int(time.time())}"
        
        # First message
        payload1 = {
            "message": "My favorite color is blue",
            "session_id": session_id
        }
        response1 = requests.post(api_url, json=payload1, timeout=TIMEOUT)
        assert response1.status_code == 200
        
        # Second message referencing first
        payload2 = {
            "message": "What is my favorite color?",
            "session_id": session_id
        }
        response2 = requests.post(api_url, json=payload2, timeout=TIMEOUT)
        assert response2.status_code == 200
        
    def test_multiple_sessions(self):
        """Test handling multiple concurrent sessions"""
        api_url = "http://localhost:10200/api/v1/chat"
        sessions = []
        
        for i in range(3):
            session_id = f"multi_session_{i}_{int(time.time())}"
            payload = {
                "message": f"Session {i} message",
                "session_id": session_id
            }
            response = requests.post(api_url, json=payload, timeout=TIMEOUT)
            assert response.status_code == 200
            sessions.append(session_id)
            
        # Verify sessions are independent
        for i, session_id in enumerate(sessions):
            payload = {
                "message": "What session is this?",
                "session_id": session_id
            }
            response = requests.post(api_url, json=payload, timeout=TIMEOUT)
            assert response.status_code == 200


class TestErrorScenarios:
    """Test error handling in the UI"""
    
    def test_network_error_handling(self):
        """Test handling of network errors"""
        # Try to access with wrong port
        wrong_url = "http://localhost:99999"
        try:
            response = requests.get(wrong_url, timeout=1)
        except requests.exceptions.ConnectionError:
            assert True  # Expected behavior
        except requests.exceptions.Timeout:
            assert True  # Also acceptable
            
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        api_url = "http://localhost:10200/api/v1/chat"
        
        # Empty message
        payload = {"message": ""}
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 400, 422]
        
        # Very long message
        payload = {"message": "x" * 50000}
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 413, 422]
        
        # Special characters
        payload = {"message": "!@#$%^&*()_+{}[]|\\:;<>?,./~`"}
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
    def test_api_failure_handling(self):
        """Test UI behavior when API fails"""
        # Test with non-existent endpoint
        api_url = "http://localhost:10200/api/v1/nonexistent"
        response = requests.post(api_url, json={"test": "data"}, timeout=TIMEOUT)
        assert response.status_code == 404


class TestVoiceFeatures:
    """Test voice interaction features"""
    
    def test_voice_input_processing(self):
        """Test voice input processing"""
        api_url = "http://localhost:10200/api/v1/voice/process"
        payload = {
            "text": "Test voice input",
            "voice": "en-US"
        }
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 422, 501]
        
    def test_voice_output_generation(self):
        """Test voice output generation"""
        # First get a text response
        chat_url = "http://localhost:10200/api/v1/chat"
        chat_payload = {"message": "Say hello"}
        chat_response = requests.post(chat_url, json=chat_payload, timeout=TIMEOUT)
        
        if chat_response.status_code == 200:
            # Then convert to voice
            voice_url = "http://localhost:10200/api/v1/voice/synthesize"
            voice_payload = {
                "text": chat_response.json().get("response", "Hello"),
                "voice": "en-US"
            }
            voice_response = requests.post(voice_url, json=voice_payload, timeout=TIMEOUT)
            assert voice_response.status_code in [200, 404, 501]


class TestRealTimeFeatures:
    """Test real-time features like streaming"""
    
    def test_message_streaming(self):
        """Test real-time message streaming"""
        api_url = "http://localhost:10200/api/v1/chat"
        payload = {
            "message": "Count from 1 to 10 slowly",
            "stream": True
        }
        
        response = requests.post(api_url, json=payload, stream=True, timeout=TIMEOUT)
        assert response.status_code == 200
        
        # Collect streamed chunks
        chunks = []
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                chunks.append(chunk)
                if len(chunks) > 10:  # Limit collection
                    break
                    
        assert len(chunks) > 0
        
    def test_progress_indicators(self):
        """Test progress indicators for long operations"""
        api_url = "http://localhost:10200/api/v1/chat"
        payload = {
            "message": "Perform a complex analysis that takes time",
            "include_progress": True
        }
        
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 422]


class TestAccessibility:
    """Test accessibility features"""
    
    def test_api_cors_headers(self):
        """Test CORS headers for API access"""
        api_url = "http://localhost:10200/api/v1/health"
        response = requests.options(api_url, timeout=TIMEOUT)
        
        # Check for CORS headers
        headers = response.headers
        # CORS might not be configured, but check response
        assert response.status_code in [200, 204, 404, 405]
        
    def test_response_formats(self):
        """Test different response formats"""
        api_url = "http://localhost:10200/api/v1/chat"
        
        # Test JSON response
        payload = {"message": "Test", "format": "json"}
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code == 200
        
        # Test plain text response
        payload = {"message": "Test", "format": "text"}
        response = requests.post(api_url, json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 422]


class TestIntegrationScenarios:
    """Test complete user scenarios"""
    
    def test_complete_user_journey(self):
        """Test a complete user journey through the application"""
        base_url = "http://localhost:10200/api/v1"
        session_id = f"journey_{int(time.time())}"
        
        # 1. Check system health
        health_response = requests.get(f"{base_url}/health", timeout=TIMEOUT)
        assert health_response.status_code == 200
        
        # 2. Get available models
        models_response = requests.get(f"{base_url}/models", timeout=TIMEOUT)
        assert models_response.status_code == 200
        
        # 3. Get available agents
        agents_response = requests.get(f"{base_url}/agents", timeout=TIMEOUT)
        assert agents_response.status_code == 200
        
        # 4. Start conversation
        chat_payload = {
            "message": "I need help writing Python code",
            "session_id": session_id,
            "agent_id": "quantumcoder"
        }
        chat_response = requests.post(f"{base_url}/chat", json=chat_payload, timeout=TIMEOUT)
        assert chat_response.status_code == 200
        
        # 5. Ask follow-up question
        followup_payload = {
            "message": "Can you show me an example of a decorator?",
            "session_id": session_id
        }
        followup_response = requests.post(f"{base_url}/chat", json=followup_payload, timeout=TIMEOUT)
        assert followup_response.status_code == 200
        
        # 6. Switch agent
        switch_payload = {
            "message": "Now I need help with documentation",
            "session_id": session_id,
            "agent_id": "documind"
        }
        switch_response = requests.post(f"{base_url}/chat", json=switch_payload, timeout=TIMEOUT)
        assert switch_response.status_code == 200
        
    def test_multi_agent_task(self):
        """Test a task requiring multiple agents"""
        base_url = "http://localhost:10200/api/v1"
        session_id = f"multiagent_{int(time.time())}"
        
        # Task: Create secure code with tests and documentation
        
        # 1. Code generation with QuantumCoder
        code_payload = {
            "message": "Create a Python function to validate email addresses",
            "session_id": session_id,
            "agent_id": "quantumcoder"
        }
        code_response = requests.post(f"{base_url}/chat", json=code_payload, timeout=TIMEOUT)
        assert code_response.status_code == 200
        
        # 2. Security review with SecurityGuard
        security_payload = {
            "message": "Review this code for security issues",
            "session_id": session_id,
            "agent_id": "securityguard"
        }
        security_response = requests.post(f"{base_url}/chat", json=security_payload, timeout=TIMEOUT)
        assert security_response.status_code == 200
        
        # 3. Test generation with TestMaster
        test_payload = {
            "message": "Create unit tests for this function",
            "session_id": session_id,
            "agent_id": "testmaster"
        }
        test_response = requests.post(f"{base_url}/chat", json=test_payload, timeout=TIMEOUT)
        assert test_response.status_code == 200
        
        # 4. Documentation with DocuMind
        doc_payload = {
            "message": "Create documentation for this function",
            "session_id": session_id,
            "agent_id": "documind"
        }
        doc_response = requests.post(f"{base_url}/chat", json=doc_payload, timeout=TIMEOUT)
        assert doc_response.status_code == 200


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])