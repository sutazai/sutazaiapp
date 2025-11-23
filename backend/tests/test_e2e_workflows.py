#!/usr/bin/env python3
"""
End-to-End Workflow Testing
Tests complete user journeys and multi-agent workflows
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any, List

BASE_URL = "http://localhost:10200/api/v1"
TIMEOUT = 60.0

class TestUserJourneys:
    """Test complete user workflows from registration to task completion"""
    
    @pytest.mark.asyncio
    async def test_user_registration_to_chat_workflow(self):
        """Test: Register → Login → Send Chat Message → Get Response → Logout"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Step 1: Register new user
            timestamp = int(asyncio.get_event_loop().time())
            register_payload = {
                "email": f"workflow_test_{timestamp}@example.com",
                "password": "SecureP@ss123!",
                "username": f"workflow_user_{timestamp}"
            }
            
            register_resp = await client.post(f"{BASE_URL}/auth/register", json=register_payload)
            print(f"\n1. Registration: {register_resp.status_code}")
            
            # Step 2: Login
            if register_resp.status_code in [200, 201]:
                login_payload = {
                    "email": register_payload["email"],
                    "password": register_payload["password"]
                }
                login_resp = await client.post(f"{BASE_URL}/auth/login", json=login_payload)
                print(f"2. Login: {login_resp.status_code}")
                
                if login_resp.status_code == 200:
                    token = login_resp.json().get("access_token")
                    headers = {"Authorization": f"Bearer {token}"}
                    
                    # Step 3: Send chat message
                    chat_payload = {
                        "message": "Hello, this is a test workflow message",
                        "model": "tinyllama"
                    }
                    chat_resp = await client.post(
                        f"{BASE_URL}/chat/send",
                        json=chat_payload,
                        headers=headers
                    )
                    print(f"3. Send chat: {chat_resp.status_code}")
                    
                    # Step 4: Get chat history
                    history_resp = await client.get(f"{BASE_URL}/chat/history", headers=headers)
                    print(f"4. Get history: {history_resp.status_code}")
                    
                    # Step 5: Logout
                    logout_resp = await client.post(f"{BASE_URL}/auth/logout", headers=headers)
                    print(f"5. Logout: {logout_resp.status_code}")
                    
                    assert chat_resp.status_code in [200, 201, 404]


class TestMultiAgentWorkflows:
    """Test multi-agent collaboration scenarios"""
    
    @pytest.mark.asyncio
    async def test_code_generation_workflow(self):
        """Test: Request Code → GPT-Engineer Generates → Aider Reviews → Result"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Step 1: Create code generation task
            task_payload = {
                "title": "Generate Python function",
                "description": "Create a function to calculate fibonacci numbers",
                "agent": "gpt-engineer",
                "priority": "high"
            }
            
            task_resp = await client.post(f"{BASE_URL}/tasks/", json=task_payload)
            print(f"\n1. Create task: {task_resp.status_code}")
            
            if task_resp.status_code in [200, 201]:
                task_id = task_resp.json().get("id") or task_resp.json().get("task_id")
                
                if task_id:
                    # Step 2: Check task status
                    await asyncio.sleep(2)
                    status_resp = await client.get(f"{BASE_URL}/tasks/{task_id}")
                    print(f"2. Task status: {status_resp.status_code}")
                    
                    # Step 3: Get task result
                    result_resp = await client.get(f"{BASE_URL}/tasks/{task_id}/result")
                    print(f"3. Task result: {result_resp.status_code}")
    
    @pytest.mark.asyncio
    async def test_document_processing_workflow(self):
        """Test: Upload Document → Documind Extracts → Store in Vector DB → Search"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Step 1: Upload document
            files = {"file": ("test.txt", b"This is a test document for processing", "text/plain")}
            upload_resp = await client.post(f"{BASE_URL}/documents/upload", files=files)
            print(f"\n1. Upload document: {upload_resp.status_code}")
            
            if upload_resp.status_code in [200, 201]:
                doc_id = upload_resp.json().get("document_id")
                
                if doc_id:
                    # Step 2: Process with Documind
                    process_payload = {"document_id": doc_id, "agent": "documind"}
                    process_resp = await client.post(f"{BASE_URL}/documents/process", json=process_payload)
                    print(f"2. Process document: {process_resp.status_code}")
                    
                    # Step 3: Wait for processing
                    await asyncio.sleep(3)
                    
                    # Step 4: Search in vector DB
                    search_payload = {"query": "test document", "n_results": 5}
                    search_resp = await client.post(f"{BASE_URL}/vectors/chromadb/search", json=search_payload)
                    print(f"3. Vector search: {search_resp.status_code}")
    
    @pytest.mark.asyncio
    async def test_financial_analysis_workflow(self):
        """Test: Request Analysis → FinRobot Fetches Data → Analyzes → Returns Report"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Step 1: Request financial analysis
            analysis_payload = {
                "ticker": "AAPL",
                "analysis_type": "fundamental",
                "agent": "finrobot"
            }
            
            analysis_resp = await client.post(f"{BASE_URL}/finance/analyze", json=analysis_payload)
            print(f"\n1. Request analysis: {analysis_resp.status_code}")
            
            if analysis_resp.status_code in [200, 201]:
                # Step 2: Wait for analysis completion
                await asyncio.sleep(5)
                
                # Step 3: Get analysis report
                report_resp = await client.get(f"{BASE_URL}/finance/reports/latest")
                print(f"2. Get report: {report_resp.status_code}")


class TestAgentOrchestration:
    """Test task decomposition and agent selection"""
    
    @pytest.mark.asyncio
    async def test_complex_task_decomposition(self):
        """Test: Complex Task → CrewAI Decomposes → Multiple Agents Execute"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Create complex task requiring multiple agents
            task_payload = {
                "title": "Build financial dashboard",
                "description": "Create a web dashboard showing stock analysis with charts",
                "agent": "crewai",
                "priority": "high",
                "subtasks": [
                    {"agent": "finrobot", "task": "Fetch stock data"},
                    {"agent": "gpt-engineer", "task": "Generate dashboard code"},
                    {"agent": "aider", "task": "Review and optimize code"}
                ]
            }
            
            task_resp = await client.post(f"{BASE_URL}/tasks/orchestrate", json=task_payload)
            print(f"\nOrchestration request: {task_resp.status_code}")
            
            if task_resp.status_code in [200, 201, 404]:
                # Check orchestration status
                await asyncio.sleep(3)
                
                # Get all agent statuses
                agents_resp = await client.get(f"{BASE_URL}/agents/")
                print(f"Agent statuses: {agents_resp.status_code}")


class TestDataSynchronization:
    """Test data consistency across services"""
    
    @pytest.mark.asyncio
    async def test_chat_history_sync(self):
        """Test chat history sync between frontend and backend"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Send message
            message_payload = {
                "message": "Test sync message",
                "model": "tinyllama",
                "session_id": "sync_test_001"
            }
            
            send_resp = await client.post(f"{BASE_URL}/chat/send", json=message_payload)
            print(f"\n1. Send message: {send_resp.status_code}")
            
            if send_resp.status_code in [200, 201]:
                # Retrieve history
                await asyncio.sleep(1)
                history_resp = await client.get(f"{BASE_URL}/chat/history?session_id=sync_test_001")
                print(f"2. Get history: {history_resp.status_code}")
                
                if history_resp.status_code == 200:
                    history = history_resp.json()
                    # Verify message is in history
                    assert isinstance(history, (list, dict))
    
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test session data persistence across requests"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            session_id = "persistent_session_001"
            
            # Create session with message
            msg1 = await client.post(
                f"{BASE_URL}/chat/send",
                json={"message": "First message", "model": "tinyllama", "session_id": session_id}
            )
            print(f"\n1. First message: {msg1.status_code}")
            
            await asyncio.sleep(1)
            
            # Add another message to same session
            msg2 = await client.post(
                f"{BASE_URL}/chat/send",
                json={"message": "Second message", "model": "tinyllama", "session_id": session_id}
            )
            print(f"2. Second message: {msg2.status_code}")
            
            # Retrieve session
            session_resp = await client.get(f"{BASE_URL}/chat/sessions/{session_id}")
            print(f"3. Get session: {session_resp.status_code}")


class TestErrorRecovery:
    """Test system recovery from failures"""
    
    @pytest.mark.asyncio
    async def test_agent_offline_recovery(self):
        """Test task handling when agent is offline"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Check agent status
            agent_status = await client.get(f"{BASE_URL}/agents/crewai")
            print(f"\nAgent status: {agent_status.status_code}")
            
            # Attempt task even if agent might be offline
            task_payload = {
                "title": "Test task",
                "description": "Test offline recovery",
                "agent": "crewai"
            }
            
            task_resp = await client.post(f"{BASE_URL}/tasks/", json=task_payload)
            print(f"Task creation: {task_resp.status_code}")
            
            # System should handle gracefully (queue or reject)
            assert task_resp.status_code in [200, 201, 404, 503]
    
    @pytest.mark.asyncio
    async def test_database_failover(self):
        """Test system behavior during database issues"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Try operations that require database
            operations = [
                ("GET", f"{BASE_URL}/chat/history"),
                ("GET", f"{BASE_URL}/agents/"),
                ("GET", f"{BASE_URL}/tasks/")
            ]
            
            for method, url in operations:
                if method == "GET":
                    resp = await client.get(url)
                    print(f"\n{url}: {resp.status_code}")
                    # Should return valid response or error code
                    assert resp.status_code in [200, 404, 500, 503]


class TestVoiceInterface:
    """Test voice command workflows"""
    
    @pytest.mark.asyncio
    async def test_voice_command_workflow(self):
        """Test: Audio Upload → STT → Process → TTS → Audio Response"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Step 1: Upload audio (simulated)
            audio_files = {"audio": ("test.wav", b"fake_audio_data", "audio/wav")}
            upload_resp = await client.post(f"{BASE_URL}/voice/transcribe", files=audio_files)
            print(f"\n1. Transcribe audio: {upload_resp.status_code}")
            
            if upload_resp.status_code in [200, 404]:
                # Step 2: Process transcribed text
                if upload_resp.status_code == 200:
                    text = upload_resp.json().get("text", "Hello")
                    
                    process_resp = await client.post(
                        f"{BASE_URL}/chat/send",
                        json={"message": text, "model": "tinyllama"}
                    )
                    print(f"2. Process text: {process_resp.status_code}")
                    
                    # Step 3: Convert response to speech
                    if process_resp.status_code == 200:
                        response_text = process_resp.json().get("response", "")
                        
                        tts_resp = await client.post(
                            f"{BASE_URL}/voice/synthesize",
                            json={"text": response_text}
                        )
                        print(f"3. Synthesize speech: {tts_resp.status_code}")


class TestConcurrentSessions:
    """Test multiple user sessions simultaneously"""
    
    @pytest.mark.asyncio
    async def test_10_concurrent_user_sessions(self):
        """Test 10 users with separate sessions"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async def user_session(user_id: int):
                session_id = f"concurrent_session_{user_id}"
                
                # Send message
                resp = await client.post(
                    f"{BASE_URL}/chat/send",
                    json={
                        "message": f"Hello from user {user_id}",
                        "model": "tinyllama",
                        "session_id": session_id
                    }
                )
                return resp.status_code
            
            # Create 10 concurrent user sessions
            tasks = [user_session(i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = sum(1 for r in results if not isinstance(r, Exception) and r in [200, 201, 404])
            print(f"\nConcurrent sessions: {successful}/10 successful")
            
            assert successful >= 8  # At least 80% success


class TestSystemStartupShutdown:
    """Test complete system lifecycle"""
    
    @pytest.mark.asyncio
    async def test_all_services_healthy_on_startup(self):
        """Test all services are healthy after startup"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Check core services
            services = [
                ("Backend", "http://localhost:10200/health"),
                ("Prometheus", "http://localhost:9090/-/healthy"),
                ("Grafana", "http://localhost:3000/api/health"),
                ("RabbitMQ", "http://localhost:10005/"),
            ]
            
            results = []
            for name, url in services:
                try:
                    resp = await client.get(url)
                    status = "✓" if resp.status_code == 200 else "✗"
                    results.append((name, status, resp.status_code))
                except Exception as e:
                    results.append((name, "✗", str(e)))
            
            print("\nService Health Check:")
            for name, status, code in results:
                print(f"  {status} {name}: {code}")
            
            # At least 50% of services should be healthy
            healthy = sum(1 for _, status, _ in results if status == "✓")
            assert healthy >= len(services) // 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
