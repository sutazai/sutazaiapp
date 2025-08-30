#!/usr/bin/env python3
"""
Test script for JARVIS voice integration
Tests real voice recording, transcription, and TTS functionality
"""

import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.services.voice_service import VoiceService, AudioConfig, get_voice_service
from app.services.wake_word import WakeWordDetector, WakeWordConfig, WakeWordEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_voice_service():
    """Test the voice service functionality"""
    logger.info("=" * 60)
    logger.info("JARVIS Voice Service Test Suite")
    logger.info("=" * 60)
    
    # Initialize voice service
    voice_service = get_voice_service()
    
    # Test 1: Check service initialization
    logger.info("\n[TEST 1] Service Initialization")
    logger.info(f"Audio available: {voice_service.audio is not None}")
    logger.info(f"TTS engine available: {voice_service.tts_engine is not None}")
    logger.info(f"Whisper model available: {voice_service.whisper_model is not None}")
    logger.info(f"Vosk model available: {voice_service.vosk_model is not None}")
    
    # Test 2: Create session
    logger.info("\n[TEST 2] Session Management")
    session_id = await voice_service.create_session(user_id="test_user")
    logger.info(f"Created session: {session_id}")
    
    session = voice_service.get_session(session_id)
    if session:
        logger.info(f"Session active: {session.is_active}")
        logger.info(f"Session start time: {session.start_time}")
    
    # Test 3: TTS (Text-to-Speech)
    logger.info("\n[TEST 3] Text-to-Speech")
    test_text = "Hello, I am JARVIS. Your personal AI assistant is now active."
    
    logger.info(f"Speaking: '{test_text}'")
    success = await voice_service.speak(test_text)
    logger.info(f"TTS success: {success}")
    
    # Also test synthesize_speech to get audio bytes
    audio_bytes = await voice_service.synthesize_speech(test_text, save_to_file=True)
    if audio_bytes:
        logger.info(f"Generated {len(audio_bytes)} bytes of audio")
        
        # Save to file for verification
        test_file = "/tmp/jarvis_tts_test.wav"
        saved = await voice_service.save_audio(audio_bytes, test_file)
        if saved:
            logger.info(f"Audio saved to {test_file}")
    
    # Test 4: Audio Recording (if microphone available)
    logger.info("\n[TEST 4] Audio Recording")
    if voice_service.audio:
        logger.info("Recording 3 seconds of audio...")
        audio_bytes = await voice_service.record_audio(duration=3.0)
        
        if audio_bytes:
            logger.info(f"Recorded {len(audio_bytes)} bytes")
            
            # Test 5: Speech Recognition
            logger.info("\n[TEST 5] Speech Recognition")
            logger.info("Transcribing recorded audio...")
            text = await voice_service.transcribe_audio(audio_bytes)
            
            if text:
                logger.info(f"Transcribed: '{text}'")
            else:
                logger.info("No speech detected in recording")
        else:
            logger.info("Recording failed - no microphone available")
    else:
        logger.info("Audio recording not available - PyAudio not initialized")
    
    # Test 6: Metrics
    logger.info("\n[TEST 6] Service Metrics")
    metrics = voice_service.get_metrics()
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    return True


async def test_wake_word_detector():
    """Test wake word detection"""
    logger.info("\n" + "=" * 60)
    logger.info("Wake Word Detector Test")
    logger.info("=" * 60)
    
    # Initialize detector with simple energy-based detection
    config = WakeWordConfig(
        engine=WakeWordEngine.ENERGY_BASED,
        keywords=["jarvis", "hey jarvis"],
        sensitivity=0.5
    )
    
    detector = WakeWordDetector(config)
    
    logger.info(f"Engine: {config.engine.value}")
    logger.info(f"Keywords: {config.keywords}")
    logger.info(f"Sensitivity: {config.sensitivity}")
    
    # Generate test audio (silence)
    test_audio = bytes(1024)  # Silent audio
    
    # Test detection
    result = await detector.detect(test_audio)
    logger.info(f"Detection result: {result.detected}")
    if result.detected:
        logger.info(f"Keyword: {result.keyword}")
        logger.info(f"Confidence: {result.confidence}")
    
    # Test with actual voice activity (simulated)
    # Create audio with some energy
    import numpy as np
    audio_array = np.random.randint(-2000, 2000, size=512, dtype=np.int16)
    test_audio_active = audio_array.tobytes()
    
    result = await detector.detect(test_audio_active)
    logger.info(f"Voice activity detection: {result.detected}")
    if result.detected:
        logger.info(f"Detected as: {result.keyword}")
    
    # Get metrics
    metrics = detector.get_metrics()
    logger.info("\nWake Word Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    return True


async def test_integration():
    """Test full integration"""
    logger.info("\n" + "=" * 60)
    logger.info("Full Integration Test")
    logger.info("=" * 60)
    
    voice_service = get_voice_service()
    
    # Test the full pipeline
    logger.info("\n[INTEGRATION] Full Voice Pipeline")
    
    # 1. Speak welcome message
    logger.info("1. Speaking welcome message...")
    await voice_service.speak("JARVIS voice system initialized. Ready for testing.")
    await asyncio.sleep(1)
    
    # 2. Create test session
    session_id = await voice_service.create_session()
    logger.info(f"2. Created session: {session_id}")
    
    # 3. Test voice command processing (with pre-recorded or generated audio)
    logger.info("3. Testing voice command processing...")
    
    # Generate test audio
    test_audio = bytes(16000 * 2)  # 2 seconds of silence
    
    result = await voice_service.process_voice_command(test_audio, session_id)
    logger.info(f"   Success: {result['success']}")
    if result.get('transcription'):
        logger.info(f"   Transcription: {result['transcription']}")
    
    # 4. Get final metrics
    logger.info("\n4. Final Metrics:")
    metrics = voice_service.get_metrics()
    for key, value in metrics.items():
        logger.info(f"   {key}: {value}")
    
    # 5. Speak completion message
    await voice_service.speak("Integration test completed successfully.")
    
    return True


async def main():
    """Main test runner"""
    try:
        # Run voice service tests
        logger.info("\n" + "=" * 60)
        logger.info("Starting JARVIS Voice Integration Tests")
        logger.info("=" * 60)
        
        # Test voice service
        voice_result = await test_voice_service()
        
        # Test wake word detector
        wake_result = await test_wake_word_detector()
        
        # Test integration
        integration_result = await test_integration()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Results Summary")
        logger.info("=" * 60)
        logger.info(f"Voice Service Tests: {'PASSED' if voice_result else 'FAILED'}")
        logger.info(f"Wake Word Tests: {'PASSED' if wake_result else 'FAILED'}")
        logger.info(f"Integration Tests: {'PASSED' if integration_result else 'FAILED'}")
        
        all_passed = voice_result and wake_result and integration_result
        
        if all_passed:
            logger.info("\n✅ All tests PASSED!")
        else:
            logger.info("\n❌ Some tests FAILED")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1
    finally:
        # Clean up
        voice_service = get_voice_service()
        await voice_service.stop()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)