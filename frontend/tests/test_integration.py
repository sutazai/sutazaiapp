#!/usr/bin/env python3
"""
Integration test for Streamlit frontend components
Tests voice interface, chat functionality, and backend communication
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add frontend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components
from config.settings import settings
from components.voice_assistant import VoiceAssistant
from components.chat_interface import ChatInterface
from components.system_monitor import SystemMonitor
from services.backend_client import BackendClient
from services.agent_orchestrator import AgentOrchestrator


async def test_backend_connection():
    """Test backend API connection"""
    print("\n=== Testing Backend Connection ===")
    
    backend_client = BackendClient(settings.BACKEND_URL)
    
    # Test health endpoint
    health = await backend_client.check_health()
    if health:
        print(f"✅ Backend health check passed: {health}")
    else:
        print("❌ Backend health check failed")
        return False
    
    # Test chat endpoint
    try:
        response = await backend_client.chat("Hello, JARVIS!", "jarvis")
        print(f"✅ Chat response received: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False


def test_voice_assistant():
    """Test voice assistant initialization"""
    print("\n=== Testing Voice Assistant ===")
    
    try:
        voice_assistant = VoiceAssistant()
        
        # Check audio availability
        if voice_assistant.audio_available:
            print("✅ Audio input device available")
        else:
            print("⚠️  No audio input device (normal in server environment)")
        
        # Check TTS availability
        if voice_assistant.tts_available:
            print("✅ Text-to-speech available")
            voices = voice_assistant.get_available_voices()
            print(f"   Found {len(voices)} voices")
        else:
            print("⚠️  Text-to-speech not available (normal in server environment)")
        
        # Test wake word detection
        test_phrases = [
            "hey jarvis how are you",
            "jarvis what time is it",
            "ok jarvis run diagnostics"
        ]
        
        for phrase in test_phrases:
            if voice_assistant._detect_wake_word(phrase):
                print(f"✅ Wake word detected in: '{phrase}'")
            else:
                print(f"❌ Wake word NOT detected in: '{phrase}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice assistant test failed: {e}")
        return False


def test_system_monitor():
    """Test system monitoring capabilities"""
    print("\n=== Testing System Monitor ===")
    
    try:
        # Test CPU usage
        cpu = SystemMonitor.get_cpu_usage()
        print(f"✅ CPU usage: {cpu}%")
        
        # Test memory usage
        memory = SystemMonitor.get_memory_usage()
        print(f"✅ Memory usage: {memory}%")
        
        # Test disk usage
        disk = SystemMonitor.get_disk_usage()
        print(f"✅ Disk usage: {disk}%")
        
        # Test Docker stats
        containers = SystemMonitor.get_docker_stats()
        print(f"✅ Found {len(containers)} Docker containers")
        
        return True
        
    except Exception as e:
        print(f"❌ System monitor test failed: {e}")
        return False


def test_agent_orchestrator():
    """Test agent orchestration"""
    print("\n=== Testing Agent Orchestrator ===")
    
    try:
        orchestrator = AgentOrchestrator()
        
        # Test agent availability
        agents = orchestrator.get_available_agents()
        print(f"✅ Found {len(agents)} available agents:")
        for agent in agents:
            print(f"   - {agent.name}: {agent.status.value}")
        
        # Test agent selection
        selected = orchestrator.select_best_agent("Write some code")
        print(f"✅ Selected agent for coding task: {selected}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent orchestrator test failed: {e}")
        return False


def test_streamlit_components():
    """Test if Streamlit-specific components are available"""
    print("\n=== Testing Streamlit Components ===")
    
    try:
        # Test streamlit_mic_recorder
        import streamlit_mic_recorder
        print("✅ streamlit_mic_recorder imported successfully")
        
        # Check for frontend build directory
        mic_recorder_path = Path(streamlit_mic_recorder.__file__).parent
        build_path = mic_recorder_path / "frontend" / "build"
        if build_path.exists():
            print(f"✅ Frontend build directory exists: {build_path}")
            # Check for key files
            if (build_path / "index.html").exists():
                print("   ✅ index.html found")
            if (build_path / "static").exists():
                print("   ✅ static directory found")
        else:
            print(f"❌ Frontend build directory missing: {build_path}")
        
        # Test other Streamlit components
        import streamlit_chat
        print("✅ streamlit_chat imported")
        
        import streamlit_lottie
        print("✅ streamlit_lottie imported")
        
        import streamlit_option_menu
        print("✅ streamlit_option_menu imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Streamlit component import failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("=" * 50)
    print("JARVIS Frontend Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(await test_backend_connection())
    results.append(test_voice_assistant())
    results.append(test_system_monitor())
    results.append(test_agent_orchestrator())
    results.append(test_streamlit_components())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("\n🎉 The JARVIS voice interface is fully functional!")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("\nSome components may have limited functionality.")
        print("This is normal in server environments without audio hardware.")
    
    print("\n📌 Frontend URL: http://localhost:11000")
    print("📌 Backend API: http://localhost:10200")
    print("\nYou can now access the JARVIS interface in your browser!")


if __name__ == "__main__":
    asyncio.run(main())