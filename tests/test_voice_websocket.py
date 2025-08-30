#!/usr/bin/env python3
"""
Test WebSocket voice streaming with JARVIS
Demonstrates real-time bidirectional audio streaming
"""

import asyncio
import base64
import json
import logging
import sys
import wave
from pathlib import Path

import websockets
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_test_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate test audio data (sine wave)"""
    frequency = 440  # A4 note
    samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, samples)
    
    # Generate sine wave
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to simulate voice
    noise = np.random.normal(0, 0.01, samples)
    audio_data = audio_data + noise
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    return audio_data.tobytes()


async def test_websocket_streaming():
    """Test WebSocket voice streaming"""
    uri = "ws://localhost:10200/api/v1/voice/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to JARVIS WebSocket")
            
            # Receive initial status
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Initial status: {data}")
            
            session_id = data.get("session_id")
            logger.info(f"Session ID: {session_id}")
            
            # Test 1: Send control command to start recording
            logger.info("\n[TEST 1] Starting recording")
            await websocket.send(json.dumps({
                "type": "control",
                "command": "start"
            }))
            
            response = await websocket.recv()
            logger.info(f"Start response: {json.loads(response)}")
            
            # Test 2: Send audio chunks
            logger.info("\n[TEST 2] Sending audio data")
            test_audio = await generate_test_audio()
            
            # Split audio into chunks
            chunk_size = 4096
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i+chunk_size]
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_base64
                }))
                
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.05)
            
            logger.info(f"Sent {len(test_audio)} bytes of audio")
            
            # Test 3: Stop recording and get transcription
            logger.info("\n[TEST 3] Stopping recording")
            await websocket.send(json.dumps({
                "type": "control",
                "command": "stop"
            }))
            
            # Wait for transcription and response
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data["type"] == "transcription":
                        logger.info(f"Transcription: {data.get('text', 'None')}")
                    elif data["type"] == "response":
                        logger.info(f"JARVIS Response: {data.get('text', 'None')}")
                        if data.get("audio"):
                            logger.info(f"Audio response received: {len(data['audio'])} characters")
                        break
                    elif data["type"] == "error":
                        logger.error(f"Error: {data.get('message')}")
                        break
                    elif data["type"] == "wake_word":
                        logger.info(f"Wake word detected: {data.get('keyword')} (confidence: {data.get('confidence')})")
                    else:
                        logger.info(f"Other response: {data}")
                        
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for response")
                    break
            
            # Test 4: Direct text input
            logger.info("\n[TEST 4] Sending direct text")
            await websocket.send(json.dumps({
                "type": "text",
                "text": "Hello JARVIS, what is your status?"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            if data["type"] == "response":
                logger.info(f"Text response: {data.get('text', 'None')}")
            
            logger.info("\n✅ WebSocket streaming test completed")
            
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket error: {e}")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def test_wake_word_streaming():
    """Test wake word detection via WebSocket"""
    uri = "ws://localhost:10200/api/v1/voice/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("\n[WAKE WORD TEST] Connected to JARVIS")
            
            # Receive initial status
            response = await websocket.recv()
            data = json.loads(response)
            session_id = data.get("session_id")
            
            # Start recording
            await websocket.send(json.dumps({
                "type": "control",
                "command": "start"
            }))
            
            await websocket.recv()  # Consume start response
            
            # Send audio that might trigger wake word detection
            logger.info("Sending audio for wake word detection...")
            
            # Generate louder audio (simulating voice activity)
            samples = 16000 * 2  # 2 seconds
            audio_data = np.random.randint(-5000, 5000, size=samples, dtype=np.int16)
            audio_bytes = audio_data.tobytes()
            
            # Send in chunks
            chunk_size = 2048
            wake_detected = False
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_base64
                }))
                
                # Check for wake word detection
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(response)
                    if data["type"] == "wake_word":
                        logger.info(f"✅ Wake word detected: {data.get('keyword')}")
                        wake_detected = True
                        break
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.01)
            
            if not wake_detected:
                logger.info("No wake word detected (expected with random audio)")
            
            # Stop recording
            await websocket.send(json.dumps({
                "type": "control",
                "command": "stop"
            }))
            
            logger.info("Wake word test completed")
            
    except Exception as e:
        logger.error(f"Wake word test failed: {e}")


async def main():
    """Main test runner"""
    logger.info("=" * 60)
    logger.info("JARVIS WebSocket Voice Streaming Test")
    logger.info("=" * 60)
    
    # Check if backend is running
    import requests
    try:
        response = requests.get("http://localhost:10200/api/v1/voice/health")
        if response.status_code == 200:
            health = response.json()
            logger.info(f"Voice service status: {health['status']}")
            logger.info(f"Components: {health['components']}")
        else:
            logger.error("Backend not responding properly")
            return 1
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to backend at localhost:10200")
        logger.error("Make sure the backend is running: docker ps | grep sutazai-backend")
        return 1
    
    # Run WebSocket tests
    await test_websocket_streaming()
    await test_wake_word_streaming()
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    # Install required packages if not present
    try:
        import websockets
    except ImportError:
        logger.info("Installing websockets...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "websockets"])
        import websockets
    
    try:
        import numpy as np
    except ImportError:
        logger.info("Installing numpy...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "numpy"])
        import numpy as np
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)