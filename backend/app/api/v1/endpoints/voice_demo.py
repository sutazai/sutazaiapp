"""Demo voice processing endpoints - No authentication required"""

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import base64
import json
import logging
from datetime import datetime, timedelta
import time
import uuid
from collections import defaultdict
import asyncio

# Import real voice services from the container
from app.services.voice_service import VoiceService, AudioConfig, get_voice_service
from app.services.wake_word import WakeWordDetector, WakeWordConfig, WakeWordEngine, get_wake_word_detector
from app.services.voice_pipeline import VoicePipeline, VoiceConfig, ASRProvider, TTSProvider
from app.services.jarvis_orchestrator import JARVISOrchestrator

logger = logging.getLogger(__name__)

# Check if speech_recognition is available
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("SpeechRecognition not available - voice features limited")

router = APIRouter()

# Rate limiting implementation
class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Use IP address as client ID
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host if request.client else "unknown"

# Initialize rate limiters with different limits for different endpoints
rate_limiter_standard = RateLimiter(max_requests=30, window_seconds=60)  # 30 requests per minute
rate_limiter_heavy = RateLimiter(max_requests=10, window_seconds=60)  # 10 requests per minute for heavy operations
rate_limiter_websocket = RateLimiter(max_requests=5, window_seconds=60)  # 5 WebSocket connections per minute

# Initialize voice components (same as authenticated version)
voice_config = VoiceConfig(
    asr_provider=ASRProvider.AUTO,
    tts_provider=TTSProvider.AUTO,
    enable_wake_word=False,
    enable_interruption=True
)

# Initialize voice service (singleton)
voice_service = get_voice_service()

# Initialize wake word detector
wake_word_config = WakeWordConfig(
    engine=WakeWordEngine.ENERGY_BASED,
    keywords=["jarvis", "hey jarvis", "ok jarvis"],
    sensitivity=0.5
)
wake_word_detector = get_wake_word_detector(wake_word_config)

# Shared orchestrator instance
jarvis_config = {
    "enable_local_models": True,
    "enable_web_search": True,
    "max_context_length": 10000
}
jarvis = JARVISOrchestrator(jarvis_config)

# Demo session management (simplified, no user tracking)
demo_sessions = {}


class VoiceRequest(BaseModel):
    """Voice processing request model"""
    audio_data: str  # Base64 encoded audio
    format: str = "wav"
    language: str = "en-US"
    session_id: Optional[str] = None
    include_context: bool = True


class VoiceResponse(BaseModel):
    """Voice processing response model"""
    text: str
    response: str
    session_id: str
    status: str
    processing_time: float
    confidence: Optional[float] = None


class TTSRequest(BaseModel):
    """Text-to-speech request model"""
    text: str
    voice: str = "default"
    language: str = "en-US"
    format: str = "mp3"


@router.post("/process", response_model=VoiceResponse)
async def process_voice_demo(
    request: VoiceRequest,
    req: Request
) -> VoiceResponse:
    """
    Process voice input through the JARVIS pipeline (DEMO - No Auth)
    Rate limited to prevent abuse
    """
    # Apply rate limiting
    client_id = rate_limiter_heavy.get_client_id(req)
    if not rate_limiter_heavy.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
    
    start_time = time.time()
    
    # Create or get demo session
    session_id = request.session_id or f"demo_{uuid.uuid4().hex[:8]}"
    if session_id not in demo_sessions:
        demo_sessions[session_id] = {
            "created": datetime.utcnow(),
            "requests": 0,
            "last_active": datetime.utcnow()
        }
    
    # Update session activity
    demo_sessions[session_id]["requests"] += 1
    demo_sessions[session_id]["last_active"] = datetime.utcnow()
    
    # Clean old sessions (older than 1 hour)
    cutoff_time = datetime.utcnow() - timedelta(hours=1)
    demo_sessions_to_remove = [
        sid for sid, data in demo_sessions.items()
        if data["last_active"] < cutoff_time
    ]
    for sid in demo_sessions_to_remove:
        del demo_sessions[sid]
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Check for wake word if enabled
        if request.session_id is None:
            detection = await wake_word_detector.detect(audio_bytes)
            if detection.detected:
                logger.info(f"Wake word detected: {detection.keyword} with confidence {detection.confidence}")
        
        # Process voice command using real voice service
        result = await voice_service.process_voice_command(
            audio_bytes,
            session_id=session_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Could not process voice command")
            )
        
        recognized_text = result["transcription"]
        
        # Process through JARVIS with limited context for demo
        context = {"session_id": session_id, "demo_mode": True} if request.include_context else {}
        jarvis_response = await jarvis.process(recognized_text, context)
        
        # Synthesize response if requested
        if jarvis_response.get("response"):
            await voice_service.speak(jarvis_response["response"])
        
        processing_time = time.time() - start_time
        
        return VoiceResponse(
            text=recognized_text,
            response=jarvis_response.get("response", ""),
            session_id=session_id,
            status="success",
            processing_time=processing_time,
            confidence=jarvis_response.get("metadata", {}).get("confidence")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo voice processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice processing failed: {str(e)}"
        )


@router.get("/health")
async def voice_health_check_demo() -> Dict[str, Any]:
    """
    Check voice processing system health (DEMO endpoint)
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "demo_mode": True,
        "components": {},
        "metrics": {},
        "active_sessions": len(demo_sessions),
        "rate_limits": {
            "standard": f"{rate_limiter_standard.max_requests} req/{rate_limiter_standard.window_seconds}s",
            "heavy": f"{rate_limiter_heavy.max_requests} req/{rate_limiter_heavy.window_seconds}s",
            "websocket": f"{rate_limiter_websocket.max_requests} conn/{rate_limiter_websocket.window_seconds}s"
        }
    }
    
    # Check voice service
    try:
        voice_metrics = voice_service.get_metrics()
        health_status["components"]["voice_service"] = "healthy"
        health_status["metrics"]["voice"] = voice_metrics
    except Exception as e:
        health_status["components"]["voice_service"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check wake word detector
    try:
        wake_metrics = wake_word_detector.get_metrics()
        health_status["components"]["wake_word"] = "healthy"
        health_status["metrics"]["wake_word"] = wake_metrics
    except Exception as e:
        health_status["components"]["wake_word"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check ASR availability
    try:
        if not SR_AVAILABLE:
            health_status["components"]["asr"] = "limited - speech_recognition not available"
            health_status["status"] = "degraded"
        else:
            health_status["components"]["asr"] = "healthy"
    except Exception as e:
        health_status["components"]["asr"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check TTS
    try:
        import pyttsx3
        engine = pyttsx3.init()
        health_status["components"]["tts"] = "healthy"
    except Exception as e:
        health_status["components"]["tts"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check JARVIS orchestrator
    if jarvis and jarvis.metrics:
        health_status["components"]["jarvis"] = "healthy"
        health_status["jarvis_metrics"] = jarvis.metrics
    else:
        health_status["components"]["jarvis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return health_status


@router.websocket("/stream")
async def voice_stream_demo(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice streaming (DEMO - Rate Limited)
    
    Protocol:
    Client -> Server: {"type": "audio", "data": base64_audio}
    Client -> Server: {"type": "control", "command": "start|stop|pause"}
    Server -> Client: {"type": "transcription", "text": "..."}
    Server -> Client: {"type": "response", "text": "...", "audio": base64_audio}
    Server -> Client: {"type": "status", "message": "..."}
    """
    # Rate limiting for WebSocket connections
    client_host = websocket.client.host if websocket.client else "unknown"
    if not rate_limiter_websocket.is_allowed(client_host):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return
    
    await websocket.accept()
    
    session_id = f"demo_ws_{uuid.uuid4().hex[:8]}"
    audio_buffer = []
    is_recording = False
    message_count = 0
    max_messages = 100  # Limit messages per connection for demo
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "message": "Connected to JARVIS voice stream (DEMO)",
            "session_id": session_id,
            "demo_mode": True,
            "message_limit": max_messages
        })
        
        # Start listening for messages
        while message_count < max_messages:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=300.0  # 5 minute timeout
                )
                message_count += 1
                msg_type = message.get("type")
                
                if msg_type == "audio":
                    # Receive audio chunk
                    audio_data = base64.b64decode(message.get("data", ""))
                    
                    if is_recording:
                        audio_buffer.append(audio_data)
                        
                        # Check for wake word in real-time
                        detection = await wake_word_detector.detect(audio_data)
                        if detection.detected:
                            await websocket.send_json({
                                "type": "wake_word",
                                "keyword": detection.keyword,
                                "confidence": detection.confidence
                            })
                
                elif msg_type == "control":
                    command = message.get("command")
                    
                    if command == "start":
                        is_recording = True
                        audio_buffer = []
                        await websocket.send_json({
                            "type": "status",
                            "message": "Recording started"
                        })
                    
                    elif command == "stop" and is_recording:
                        is_recording = False
                        
                        if audio_buffer:
                            # Process accumulated audio
                            full_audio = b''.join(audio_buffer)
                            
                            # Transcribe
                            text = await voice_service.transcribe_audio(full_audio)
                            
                            if text:
                                # Send transcription
                                await websocket.send_json({
                                    "type": "transcription",
                                    "text": text
                                })
                                
                                # Process through JARVIS (with demo context)
                                context = {"session_id": session_id, "demo_mode": True}
                                jarvis_response = await jarvis.process(text, context)
                                
                                response_text = jarvis_response.get("response", "")
                                
                                # Generate TTS response
                                if response_text:
                                    audio_response = await voice_service.synthesize_speech(
                                        response_text,
                                        save_to_file=True
                                    )
                                    
                                    # Send response with audio
                                    response_data = {
                                        "type": "response",
                                        "text": response_text
                                    }
                                    
                                    if audio_response:
                                        response_data["audio"] = base64.b64encode(audio_response).decode('utf-8')
                                    
                                    await websocket.send_json(response_data)
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "Could not transcribe audio"
                                })
                        
                        audio_buffer = []
                    
                    elif command == "pause":
                        is_recording = False
                        await websocket.send_json({
                            "type": "status",
                            "message": "Recording paused"
                        })
                
                elif msg_type == "text":
                    # Direct text input (bypass ASR)
                    text = message.get("text", "")
                    
                    if text:
                        # Process through JARVIS (with demo context)
                        context = {"session_id": session_id, "demo_mode": True}
                        jarvis_response = await jarvis.process(text, context)
                        
                        response_text = jarvis_response.get("response", "")
                        
                        # Generate TTS response
                        if response_text:
                            audio_response = await voice_service.synthesize_speech(
                                response_text,
                                save_to_file=True
                            )
                            
                            response_data = {
                                "type": "response",
                                "text": response_text
                            }
                            
                            if audio_response:
                                response_data["audio"] = base64.b64encode(audio_response).decode('utf-8')
                            
                            await websocket.send_json(response_data)
                
                # Send remaining message count periodically
                if message_count % 10 == 0:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Demo mode: {max_messages - message_count} messages remaining"
                    })
                    
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Connection timeout - idle for too long"
                })
                break
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message"
                })
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
        
        # Message limit reached
        if message_count >= max_messages:
            await websocket.send_json({
                "type": "status",
                "message": "Demo session limit reached. Connection closing."
            })
            await websocket.close(code=1000, reason="Demo limit reached")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for demo session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in demo: {e}")
    finally:
        # Clean up demo session
        logger.info(f"Demo session {session_id} ended after {message_count} messages")


@router.post("/transcribe")
async def transcribe_audio_demo(
    req: Request,
    audio_file: UploadFile = File(...),
    language: str = "en"
) -> Dict[str, Any]:
    """
    Transcribe audio file to text (DEMO - Rate Limited)
    """
    # Apply rate limiting
    client_id = rate_limiter_heavy.get_client_id(req)
    if not rate_limiter_heavy.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
    
    # Limit file size for demo (10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    
    try:
        # Read audio file with size limit
        audio_content = await audio_file.read()
        
        if len(audio_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large for demo. Maximum size is {max_size // (1024*1024)}MB"
            )
        
        # Use real voice service for transcription
        text = await voice_service.transcribe_audio(audio_content, language=language)
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio"
            )
        
        return {
            "text": text,
            "language": language,
            "filename": audio_file.filename,
            "status": "success",
            "demo_mode": True,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/synthesize")
async def text_to_speech_demo(
    request: TTSRequest,
    req: Request
) -> Dict[str, Any]:
    """
    Convert text to speech (DEMO - Rate Limited)
    """
    # Apply rate limiting
    client_id = rate_limiter_standard.get_client_id(req)
    if not rate_limiter_standard.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
    
    # Limit text length for demo (500 characters)
    max_text_length = 500
    if len(request.text) > max_text_length:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long for demo. Maximum length is {max_text_length} characters"
        )
    
    try:
        # Use real voice service for TTS
        audio_bytes = await voice_service.synthesize_speech(
            text=request.text,
            voice=request.voice if request.voice != "default" else None,
            save_to_file=True
        )
        
        if audio_bytes:
            # Encode audio to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return {
                "status": "success",
                "text": request.text,
                "voice": request.voice,
                "format": request.format,
                "audio_data": audio_base64,
                "audio_size": len(audio_bytes),
                "demo_mode": True,
                "duration_estimate": len(request.text) * 0.06
            }
        else:
            return {
                "status": "partial",
                "text": request.text,
                "voice": request.voice,
                "format": request.format,
                "audio_data": None,
                "demo_mode": True,
                "message": "TTS synthesis completed but audio generation failed"
            }
        
    except Exception as e:
        logger.error(f"Demo TTS error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text-to-speech failed: {str(e)}"
        )


@router.get("/voices")
async def list_voices_demo() -> Dict[str, Any]:
    """
    List available TTS voices (DEMO)
    """
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        voice_list = []
        for voice in voices[:5]:  # Limit to first 5 voices for demo
            voice_list.append({
                "id": voice.id,
                "name": voice.name,
                "languages": getattr(voice, 'languages', ['en-US']),
                "gender": getattr(voice, 'gender', 'unknown')
            })
        
        return {
            "voices": voice_list,
            "default": "default",
            "count": len(voice_list),
            "demo_mode": True,
            "total_available": len(voices)
        }
        
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        return {
            "voices": [],
            "default": "default",
            "count": 0,
            "demo_mode": True,
            "error": str(e)
        }


@router.get("/asr-providers")
async def list_asr_providers_demo() -> Dict[str, Any]:
    """
    List available ASR providers (DEMO)
    """
    providers = []
    
    # Check Whisper
    try:
        import whisper
        providers.append({
            "name": "whisper",
            "status": "available",
            "models": ["tiny", "base"],  # Only small models for demo
            "languages": "multiple"
        })
    except ImportError:
        providers.append({
            "name": "whisper",
            "status": "not_installed"
        })
    
    # Check Vosk
    try:
        import vosk
        providers.append({
            "name": "vosk",
            "status": "available",
            "models": ["small"],  # Only small model for demo
            "languages": "multiple"
        })
    except ImportError:
        providers.append({
            "name": "vosk",
            "status": "not_installed"
        })
    
    # Google Speech API
    providers.append({
        "name": "google",
        "status": "available",
        "requires_internet": True,
        "languages": "multiple",
        "demo_limitations": "Limited requests per day"
    })
    
    return {
        "providers": providers,
        "default": "auto",
        "count": len([p for p in providers if p.get("status") == "available"]),
        "demo_mode": True
    }


@router.post("/wake-word/test")
async def test_wake_word_demo(
    req: Request,
    audio_file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Test wake word detection on an audio file (DEMO - Rate Limited)
    """
    # Apply rate limiting
    client_id = rate_limiter_standard.get_client_id(req)
    if not rate_limiter_standard.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making another request."
        )
    
    # Limit file size for demo (5MB)
    max_size = 5 * 1024 * 1024  # 5MB
    
    try:
        # Read audio file with size limit
        audio_content = await audio_file.read()
        
        if len(audio_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large for demo. Maximum size is {max_size // (1024*1024)}MB"
            )
        
        # Test wake word detection
        detection = await wake_word_detector.detect(audio_content)
        
        return {
            "detected": detection.detected,
            "keyword": detection.keyword,
            "confidence": detection.confidence,
            "filename": audio_file.filename,
            "engine": wake_word_detector.config.engine.value,
            "demo_mode": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo wake word test error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        )


@router.get("/demo-info")
async def get_demo_info() -> Dict[str, Any]:
    """
    Get information about demo limitations and features
    """
    return {
        "demo_mode": True,
        "features": {
            "voice_processing": True,
            "transcription": True,
            "text_to_speech": True,
            "wake_word_detection": True,
            "websocket_streaming": True
        },
        "limitations": {
            "rate_limits": {
                "standard_endpoints": "30 requests per minute",
                "heavy_endpoints": "10 requests per minute",
                "websocket_connections": "5 connections per minute"
            },
            "file_size_limits": {
                "audio_transcription": "10MB",
                "wake_word_test": "5MB"
            },
            "text_limits": {
                "tts_max_length": "500 characters"
            },
            "session_limits": {
                "session_duration": "1 hour",
                "websocket_messages": "100 messages per connection",
                "websocket_timeout": "5 minutes idle"
            },
            "model_restrictions": {
                "whisper": "tiny and base models only",
                "vosk": "small model only",
                "voices": "first 5 voices only"
            }
        },
        "endpoints": {
            "process": "POST /api/v1/voice/demo/process - Process voice commands",
            "health": "GET /api/v1/voice/demo/health - Health check",
            "stream": "WebSocket /api/v1/voice/demo/stream - Real-time streaming",
            "transcribe": "POST /api/v1/voice/demo/transcribe - Transcribe audio",
            "synthesize": "POST /api/v1/voice/demo/synthesize - Text to speech",
            "voices": "GET /api/v1/voice/demo/voices - List TTS voices",
            "asr_providers": "GET /api/v1/voice/demo/asr-providers - List ASR providers",
            "wake_word_test": "POST /api/v1/voice/demo/wake-word/test - Test wake word",
            "demo_info": "GET /api/v1/voice/demo/demo-info - This endpoint"
        },
        "note": "This is a demo API with rate limiting and reduced capabilities. For full access, please authenticate."
    }