#!/usr/bin/env python3
"""
Test JARVIS AI Integration
Verify that the complete system is working
"""

import requests
import json
import time
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_banner():
    """Print test banner"""
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}    JARVIS AI System Integration Test{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

def test_backend():
    """Test backend API"""
    print(f"{BLUE}Testing Backend API...{RESET}")
    
    try:
        # Test health
        response = requests.get("http://localhost:10200/health", timeout=2)
        if response.status_code == 200:
            print(f"  {GREEN}✓{RESET} Backend health check passed")
        else:
            print(f"  {RED}✗{RESET} Backend health check failed")
            return False
        
        # Test chat endpoint
        test_message = "Hello JARVIS, are you operational?"
        response = requests.post(
            "http://localhost:10200/api/v1/chat/",
            json={"message": test_message},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            ai_response = data.get('response', 'No response')
            print(f"  {GREEN}✓{RESET} Chat API working")
            print(f"    {CYAN}User:{RESET} {test_message}")
            print(f"    {MAGENTA}JARVIS:{RESET} {ai_response[:100]}...")
            return True
        else:
            print(f"  {RED}✗{RESET} Chat API failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  {RED}✗{RESET} Backend test failed: {str(e)}")
        return False

def test_frontend():
    """Test frontend UI"""
    print(f"\n{BLUE}Testing Frontend UI...{RESET}")
    
    try:
        # Test Streamlit health
        response = requests.get("http://localhost:11000/_stcore/health", timeout=2)
        if response.status_code == 200:
            print(f"  {GREEN}✓{RESET} Frontend is running on port 11000")
            return True
        else:
            print(f"  {RED}✗{RESET} Frontend health check failed")
            return False
            
    except Exception as e:
        print(f"  {RED}✗{RESET} Frontend test failed: {str(e)}")
        return False

def test_voice_service():
    """Test voice service"""
    print(f"\n{BLUE}Testing Voice Service...{RESET}")
    
    try:
        response = requests.get("http://localhost:10200/api/v1/voice/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            if status == 'healthy':
                print(f"  {GREEN}✓{RESET} Voice service is healthy")
                print(f"    Wake word detection: {data.get('wake_word_enabled', False)}")
                print(f"    TTS available: {data.get('tts_available', False)}")
                print(f"    STT available: {data.get('stt_available', False)}")
            else:
                print(f"  {YELLOW}⚠{RESET} Voice service status: {status}")
            return True
        else:
            print(f"  {YELLOW}⚠{RESET} Voice service not available (optional)")
            return True  # Voice is optional
            
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Voice service not available: {str(e)}")
        return True  # Voice is optional

def print_summary(results):
    """Print test summary"""
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}Test Summary{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = f"{GREEN}✓ PASSED{RESET}" if passed else f"{RED}✗ FAILED{RESET}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{CYAN}{'='*60}{RESET}")
    
    if all_passed:
        print(f"{GREEN}{BOLD}🎉 ALL SYSTEMS OPERATIONAL! 🎉{RESET}")
        print(f"\n{BOLD}Access JARVIS at:{RESET}")
        print(f"  {CYAN}Frontend UI:{RESET} http://localhost:11000")
        print(f"  {CYAN}Backend API:{RESET} http://localhost:10200/docs")
        print(f"\n{BOLD}Features Available:{RESET}")
        print(f"  • Text chat with AI assistant")
        print(f"  • Voice input/output (if configured)")
        print(f"  • Real-time streaming responses")
        print(f"  • WebSocket support for live updates")
        print(f"  • System monitoring dashboard")
    else:
        print(f"{YELLOW}{BOLD}⚠ SOME COMPONENTS NEED ATTENTION{RESET}")
        print(f"\nPlease check the failed components above.")
    
    print(f"{CYAN}{'='*60}{RESET}\n")

def main():
    """Run all integration tests"""
    print_banner()
    
    print(f"{BOLD}Time:{RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {
        "Backend API": test_backend(),
        "Frontend UI": test_frontend(),
        "Voice Service": test_voice_service()
    }
    
    print_summary(results)
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit(main())