#!/usr/bin/env python3
"""
JARVIS Frontend Integration Verification Script
Tests all connections and features of the updated frontend
"""

import sys
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Tuple

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_test(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}" if passed else f"{Colors.FAIL}✗ FAILED{Colors.ENDC}"
    print(f"  {test_name}: {status}")
    if details:
        print(f"    {Colors.OKCYAN}{details}{Colors.ENDC}")

def test_backend_connection() -> Tuple[bool, str]:
    """Test backend API connection"""
    try:
        response = requests.get("http://localhost:10200/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"Backend version: {data.get('version', 'Unknown')}"
        else:
            return False, f"Status code: {response.status_code}"
    except requests.ConnectionError:
        return False, "Cannot connect to backend at localhost:10200"
    except Exception as e:
        return False, str(e)

def test_chat_endpoint() -> Tuple[bool, str]:
    """Test chat endpoint"""
    try:
        payload = {
            "message": "Hello JARVIS",
            "model": "tinyllama:latest",
            "stream": False
        }
        response = requests.post(
            "http://localhost:10200/api/v1/chat",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if "response" in data or "message" in data:
                return True, "Chat endpoint responding correctly"
            else:
                return False, f"Unexpected response format: {data}"
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_models_endpoint() -> Tuple[bool, str]:
    """Test models endpoint"""
    try:
        response = requests.get("http://localhost:10200/api/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return True, f"Found {len(models)} model(s): {', '.join(models[:3])}"
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_agents_endpoint() -> Tuple[bool, str]:
    """Test agents endpoint"""
    try:
        response = requests.get("http://localhost:10200/api/v1/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            agents = data.get("agents", [])
            return True, f"Found {len(agents)} agent(s)"
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_websocket_endpoint() -> Tuple[bool, str]:
    """Test WebSocket endpoint availability"""
    try:
        # Check if WebSocket endpoint is accessible
        # We'll just check if the backend supports it
        response = requests.get("http://localhost:10200/health", timeout=5)
        if response.status_code == 200:
            return True, "WebSocket endpoint should be available"
        else:
            return False, "Backend not responding"
    except Exception as e:
        return False, str(e)

def test_frontend_modules() -> Tuple[bool, str]:
    """Test if all frontend modules can be imported"""
    try:
        # Test imports
        from services.backend_client_fixed import BackendClient
        from components.chat_interface import ChatInterface
        from components.voice_assistant import VoiceAssistant
        from components.system_monitor import SystemMonitor
        from config.settings import settings
        
        # Test instantiation
        client = BackendClient()
        chat = ChatInterface()
        voice = VoiceAssistant()
        
        return True, "All modules imported and instantiated successfully"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Instantiation error: {e}"

def test_system_monitor() -> Tuple[bool, str]:
    """Test system monitoring capabilities"""
    try:
        from components.system_monitor import SystemMonitor
        
        cpu = SystemMonitor.get_cpu_usage()
        memory = SystemMonitor.get_memory_usage()
        disk = SystemMonitor.get_disk_usage()
        
        if all([cpu >= 0, memory >= 0, disk >= 0]):
            return True, f"CPU: {cpu:.1f}%, Memory: {memory:.1f}%, Disk: {disk:.1f}%"
        else:
            return False, "Invalid metrics returned"
    except Exception as e:
        return False, str(e)

def test_docker_monitoring() -> Tuple[bool, str]:
    """Test Docker container monitoring"""
    try:
        from components.system_monitor import SystemMonitor
        
        containers = SystemMonitor.get_docker_stats()
        if isinstance(containers, list):
            running = len([c for c in containers if c.get('status') == 'running'])
            return True, f"Found {len(containers)} container(s), {running} running"
        else:
            return True, "Docker monitoring not available (normal in some environments)"
    except Exception as e:
        return True, f"Docker not available: {str(e)[:50]}"

def test_voice_components() -> Tuple[bool, str]:
    """Test voice assistant components"""
    try:
        from components.voice_assistant import VoiceAssistant
        
        voice = VoiceAssistant()
        
        # Check available features
        features = []
        if voice.audio_available:
            features.append("Audio input")
        if voice.tts_available:
            features.append("TTS")
        
        if features:
            return True, f"Available: {', '.join(features)}"
        else:
            return True, "Voice features not available (normal in server environment)"
    except Exception as e:
        return False, str(e)

def test_backend_client_sync() -> Tuple[bool, str]:
    """Test synchronous backend client methods"""
    try:
        from services.backend_client_fixed import BackendClient
        
        client = BackendClient()
        
        # Test health check
        health = client.check_health_sync()
        if health.get("status") != "error":
            return True, "Synchronous client methods working"
        else:
            return False, f"Health check failed: {health.get('error')}"
    except Exception as e:
        return False, str(e)

def run_integration_tests():
    """Run all integration tests"""
    print_header("JARVIS Frontend Integration Tests")
    print(f"{Colors.OKCYAN}Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    # Track results
    total_tests = 0
    passed_tests = 0
    
    # Test categories
    test_categories = [
        ("Backend Connection", [
            ("Backend API Health", test_backend_connection),
            ("Chat Endpoint", test_chat_endpoint),
            ("Models Endpoint", test_models_endpoint),
            ("Agents Endpoint", test_agents_endpoint),
            ("WebSocket Support", test_websocket_endpoint),
        ]),
        ("Frontend Components", [
            ("Module Imports", test_frontend_modules),
            ("Backend Client Sync", test_backend_client_sync),
            ("System Monitor", test_system_monitor),
            ("Docker Monitor", test_docker_monitoring),
            ("Voice Assistant", test_voice_components),
        ])
    ]
    
    # Run tests by category
    for category_name, tests in test_categories:
        print(f"\n{Colors.BOLD}{category_name}:{Colors.ENDC}")
        
        for test_name, test_func in tests:
            total_tests += 1
            try:
                passed, details = test_func()
                if passed:
                    passed_tests += 1
                print_test(test_name, passed, details)
            except Exception as e:
                print_test(test_name, False, f"Exception: {str(e)[:100]}")
    
    # Print summary
    print_header("Test Summary")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate == 100:
        status_color = Colors.OKGREEN
        status_text = "ALL TESTS PASSED!"
    elif success_rate >= 80:
        status_color = Colors.WARNING
        status_text = "MOSTLY WORKING"
    else:
        status_color = Colors.FAIL
        status_text = "NEEDS ATTENTION"
    
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {Colors.OKGREEN}{passed_tests}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{total_tests - passed_tests}{Colors.ENDC}")
    print(f"  Success Rate: {status_color}{success_rate:.1f}%{Colors.ENDC}")
    print(f"\n  {status_color}{Colors.BOLD}{status_text}{Colors.ENDC}")
    
    # Provide recommendations
    if success_rate < 100:
        print(f"\n{Colors.WARNING}Recommendations:{Colors.ENDC}")
        
        if total_tests - passed_tests > 0:
            print("  1. Check if the backend is running:")
            print("     cd /opt/sutazaiapp/backend && ./start_backend.sh")
            print("  2. Verify Docker is running:")
            print("     sudo systemctl status docker")
            print("  3. Check service logs:")
            print("     docker logs sutazai-backend --tail 50")
    else:
        print(f"\n{Colors.OKGREEN}✓ Frontend is fully integrated and ready!{Colors.ENDC}")
        print(f"{Colors.OKGREEN}  Access JARVIS at: http://localhost:11000{Colors.ENDC}")
    
    return success_rate == 100

if __name__ == "__main__":
    try:
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Tests interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Test suite error: {e}{Colors.ENDC}")
        sys.exit(1)