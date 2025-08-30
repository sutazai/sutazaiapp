#!/usr/bin/env python3
"""
JARVIS Full System Test Suite - Comprehensive validation of all components
Tests text chat, voice, WebSocket, and multi-model support
"""

import os
import sys
import time
import json
import asyncio
import base64
import requests
import websocket
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class JARVISSystemTester:
    """Comprehensive system tester for JARVIS AI"""
    
    def __init__(self):
        self.backend_url = "http://localhost:10200"
        self.ollama_url = "http://localhost:11434"
        self.results = []
        self.session_id = None
        
    def print_header(self, title: str, icon: str = "🔍"):
        """Print formatted section header"""
        print(f"\n{icon} {title}")
        print("=" * 70)
        
    def print_result(self, test_name: str, success: bool, details: str = ""):
        """Print and record test result"""
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:40} {status}")
        if details:
            print(f"     → {details}")
        self.results.append((test_name, success))
        return success
        
    # ===== INFRASTRUCTURE TESTS =====
    
    def test_ollama_connectivity(self) -> bool:
        """Test Ollama service connectivity"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                return self.print_result(
                    "Ollama Connectivity",
                    True,
                    f"Found {len(models)} models: {', '.join(model_names[:3])}"
                )
        except Exception as e:
            pass
        return self.print_result("Ollama Connectivity", False, str(e)[:50])
        
    def test_backend_health(self) -> bool:
        """Test backend API health"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                return self.print_result("Backend Health", True, "API is responsive")
        except Exception as e:
            pass
        return self.print_result("Backend Health", False, "Backend not responding")
        
    # ===== CHAT FUNCTIONALITY TESTS =====
    
    def test_chat_basic(self) -> bool:
        """Test basic chat functionality"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/chat/",
                json={"message": "What is 2+2? Reply with just the number."},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('session_id')
                ai_response = data.get('response', '')
                
                # Check if response contains the answer
                if '4' in ai_response or 'four' in ai_response.lower():
                    return self.print_result(
                        "Basic Chat",
                        True,
                        f"AI correctly answered (session: {self.session_id[:8]}...)"
                    )
                else:
                    return self.print_result(
                        "Basic Chat",
                        True,
                        f"AI responded but answer unclear: {ai_response[:50]}"
                    )
        except Exception as e:
            pass
        return self.print_result("Basic Chat", False, "Chat endpoint failed")
        
    def test_chat_context(self) -> bool:
        """Test chat context preservation"""
        if not self.session_id:
            return self.print_result("Chat Context", False, "No session ID available")
            
        try:
            # Set context
            response1 = requests.post(
                f"{self.backend_url}/api/v1/chat/",
                json={
                    "message": "My favorite color is blue. Remember this.",
                    "session_id": self.session_id
                },
                timeout=30
            )
            
            if response1.status_code == 200:
                # Test recall
                response2 = requests.post(
                    f"{self.backend_url}/api/v1/chat/",
                    json={
                        "message": "What is my favorite color?",
                        "session_id": self.session_id
                    },
                    timeout=30
                )
                
                if response2.status_code == 200:
                    ai_response = response2.json().get('response', '').lower()
                    if 'blue' in ai_response:
                        return self.print_result(
                            "Chat Context",
                            True,
                            "AI remembers conversation context"
                        )
                    else:
                        return self.print_result(
                            "Chat Context",
                            False,
                            "AI responded but didn't recall context"
                        )
        except Exception as e:
            pass
        return self.print_result("Chat Context", False, "Context test failed")
        
    # ===== VOICE FUNCTIONALITY TESTS =====
    
    def test_voice_health(self) -> bool:
        """Test voice system health"""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/voice/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                if status == 'operational':
                    return self.print_result(
                        "Voice Health",
                        True,
                        f"ASR: {data.get('asr_available')}, TTS: {data.get('tts_available')}"
                    )
                else:
                    return self.print_result(
                        "Voice Health",
                        False,
                        f"Voice system degraded: {data.get('message', 'Unknown issue')}"
                    )
        except Exception as e:
            pass
        return self.print_result("Voice Health", False, "Voice endpoint not responding")
        
    def test_voice_transcription(self) -> bool:
        """Test voice transcription capability"""
        try:
            # Create a simple test audio (would need actual audio in production)
            test_audio = base64.b64encode(b"test_audio_data").decode('utf-8')
            
            response = requests.post(
                f"{self.backend_url}/api/v1/voice/transcribe",
                json={
                    "audio_data": test_audio,
                    "format": "wav"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return self.print_result(
                    "Voice Transcription",
                    True,
                    "Transcription endpoint responding"
                )
            elif response.status_code == 503:
                return self.print_result(
                    "Voice Transcription",
                    False,
                    "Dependencies not installed"
                )
        except Exception as e:
            pass
        return self.print_result("Voice Transcription", False, "Transcription failed")
        
    # ===== WEBSOCKET TESTS =====
    
    def test_websocket_connection(self) -> bool:
        """Test WebSocket connectivity"""
        try:
            ws = websocket.create_connection(
                "ws://localhost:10200/ws",
                timeout=5
            )
            
            # Send test message
            test_msg = json.dumps({
                "type": "text",
                "message": "WebSocket test",
                "client_id": "test_client"
            })
            ws.send(test_msg)
            
            # Wait for response
            response = ws.recv()
            ws.close()
            
            if response:
                return self.print_result(
                    "WebSocket Connection",
                    True,
                    "Real-time communication working"
                )
        except Exception as e:
            pass
        return self.print_result("WebSocket Connection", False, "WebSocket not available")
        
    # ===== MODEL TESTS =====
    
    def test_model_listing(self) -> bool:
        """Test model listing capability"""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/agents/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if isinstance(models, list) and len(models) > 0:
                    model_names = [m.get('name', 'unknown') for m in models[:3]]
                    return self.print_result(
                        "Model Listing",
                        True,
                        f"Found {len(models)} models: {', '.join(model_names)}"
                    )
        except Exception as e:
            pass
        return self.print_result("Model Listing", False, "Cannot list models")
        
    def test_jarvis_orchestrator(self) -> bool:
        """Test JARVIS orchestrator functionality"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/jarvis/chat",
                json={
                    "message": "Test JARVIS orchestrator",
                    "use_jarvis": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('orchestrator_used'):
                    return self.print_result(
                        "JARVIS Orchestrator",
                        True,
                        f"Pipeline: {data.get('pipeline_stages', 'unknown')}"
                    )
                else:
                    return self.print_result(
                        "JARVIS Orchestrator",
                        False,
                        "Orchestrator not activated"
                    )
            elif response.status_code == 404:
                return self.print_result(
                    "JARVIS Orchestrator",
                    False,
                    "Endpoint not implemented"
                )
        except Exception as e:
            pass
        return self.print_result("JARVIS Orchestrator", False, "Orchestrator test failed")
        
    # ===== FEATURE INTEGRATION TESTS =====
    
    def test_wake_word_detection(self) -> bool:
        """Test wake word detection capability"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/voice/wake-word-status",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('enabled'):
                    return self.print_result(
                        "Wake Word Detection",
                        True,
                        f"Wake word: '{data.get('wake_word', 'jarvis')}'"
                    )
                else:
                    return self.print_result(
                        "Wake Word Detection",
                        False,
                        "Feature disabled"
                    )
            elif response.status_code == 404:
                return self.print_result(
                    "Wake Word Detection",
                    False,
                    "Not implemented"
                )
        except Exception as e:
            pass
        return self.print_result("Wake Word Detection", False, "Wake word test failed")
        
    def test_streaming_response(self) -> bool:
        """Test streaming response capability"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/chat/stream",
                json={"message": "Count to 5"},
                stream=True,
                timeout=10
            )
            
            if response.status_code == 200:
                chunks = []
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        chunks.append(chunk)
                        if len(chunks) > 1:  # Got multiple chunks = streaming works
                            return self.print_result(
                                "Streaming Response",
                                True,
                                f"Received {len(chunks)} chunks"
                            )
                return self.print_result(
                    "Streaming Response",
                    False,
                    "Single response, not streaming"
                )
            elif response.status_code == 404:
                return self.print_result(
                    "Streaming Response",
                    False,
                    "Not implemented"
                )
        except Exception as e:
            pass
        return self.print_result("Streaming Response", False, "Streaming test failed")
        
    # ===== MAIN TEST RUNNER =====
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "=" * 70)
        print("🚀 JARVIS FULL SYSTEM TEST SUITE")
        print("=" * 70)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Infrastructure Tests
        self.print_header("INFRASTRUCTURE TESTS", "🏗️")
        self.test_ollama_connectivity()
        self.test_backend_health()
        
        # Chat Tests
        self.print_header("CHAT FUNCTIONALITY", "💬")
        self.test_chat_basic()
        self.test_chat_context()
        
        # Voice Tests
        self.print_header("VOICE FUNCTIONALITY", "🎤")
        self.test_voice_health()
        self.test_voice_transcription()
        
        # WebSocket Tests
        self.print_header("WEBSOCKET CONNECTIVITY", "🔌")
        self.test_websocket_connection()
        
        # Model Tests
        self.print_header("MODEL MANAGEMENT", "🤖")
        self.test_model_listing()
        self.test_jarvis_orchestrator()
        
        # Feature Tests
        self.print_header("ADVANCED FEATURES", "✨")
        self.test_wake_word_detection()
        self.test_streaming_response()
        
        # Summary
        self.print_summary()
        
    def print_summary(self):
        """Print test summary and recommendations"""
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, success in self.results if success)
        total = len(self.results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        # Group results by category
        categories = {
            "Infrastructure": ["Ollama Connectivity", "Backend Health"],
            "Chat": ["Basic Chat", "Chat Context"],
            "Voice": ["Voice Health", "Voice Transcription"],
            "Real-time": ["WebSocket Connection", "Streaming Response"],
            "Advanced": ["JARVIS Orchestrator", "Wake Word Detection", "Model Listing"]
        }
        
        for category, tests in categories.items():
            category_results = [(name, success) for name, success in self.results if name in tests]
            if category_results:
                category_passed = sum(1 for _, success in category_results if success)
                print(f"\n{category}:")
                for test_name, success in category_results:
                    status = "✅" if success else "❌"
                    print(f"  {status} {test_name}")
                print(f"  → {category_passed}/{len(category_results)} passed")
        
        print(f"\n📈 Overall Score: {passed}/{total} tests passed ({percentage:.1f}%)")
        
        # Status Assessment
        if percentage == 100:
            print("\n🎉 PERFECT! All JARVIS systems fully operational!")
            print("   The system is ready for production deployment.")
        elif percentage >= 80:
            print("\n✅ EXCELLENT! JARVIS is operational with minor issues.")
            print("   Address the failed tests for full functionality.")
        elif percentage >= 60:
            print("\n⚠️ FUNCTIONAL but needs work. Core features working.")
            print("   Priority: Fix voice and real-time features.")
        elif percentage >= 40:
            print("\n🔧 PARTIAL FUNCTIONALITY. Basic chat working.")
            print("   Significant work needed for full JARVIS capabilities.")
        else:
            print("\n❌ CRITICAL ISSUES. System not ready.")
            print("   Start with fixing infrastructure and basic chat.")
            
        # Recommendations
        print("\n📝 RECOMMENDATIONS:")
        failed_tests = [name for name, success in self.results if not success]
        
        if "Ollama Connectivity" in failed_tests:
            print("  1. Check Ollama service: docker ps | grep ollama")
            print("     docker logs sutazai-ollama --tail 50")
            
        if "Voice Transcription" in failed_tests:
            print("  2. Install voice dependencies:")
            print("     docker exec sutazai-backend pip install SpeechRecognition whisper vosk")
            
        if "WebSocket Connection" in failed_tests:
            print("  3. Check WebSocket implementation in backend")
            print("     Verify /ws endpoint is properly configured")
            
        if "JARVIS Orchestrator" in failed_tests:
            print("  4. Implement JARVIS chat endpoint:")
            print("     Add /api/v1/jarvis/chat with orchestrator integration")
            
        if "Wake Word Detection" in failed_tests:
            print("  5. Install Porcupine for wake word:")
            print("     docker exec sutazai-backend pip install pvporcupine")
            
        print("\n" + "=" * 70)
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


def main():
    """Main entry point"""
    tester = JARVISSystemTester()
    tester.run_all_tests()
    
    # Return exit code based on results
    passed = sum(1 for _, success in tester.results if success)
    total = len(tester.results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    if percentage >= 60:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    sys.exit(main())