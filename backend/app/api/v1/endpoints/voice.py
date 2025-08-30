"""Voice processing endpoints for JARVIS"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any, Annotated, List
import base64
import io
import json
import logging
from datetime import datetime

from app.services.voice_service import VoiceService, AudioConfig, get_voice_service
from app.services.wake_word import WakeWordDetector, WakeWordConfig, WakeWordEngine, get_wake_word_detector
from app.services.voice_pipeline import VoicePipeline, VoiceConfig, ASRProvider, TTSProvider
from app.services.jarvis_orchestrator import JARVISOrchestrator
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies.auth import get_current_active_user
from app.models.user import User

logger = logging.getLogger(__name__)

# Check if speech_recognition is available
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("SpeechRecognition not available - voice features limited")

router = APIRouter()

# Initialize components
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
    engine=WakeWordEngine.ENERGY_BASED,  # Start with simple detection
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
async def process_voice(
    request: VoiceRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
) -> VoiceResponse:
    """
    Process voice input through the JARVIS pipeline
    1. Convert audio to text (ASR)
    2. Process through JARVIS orchestrator
    3. Return text response
    """
    import time
    import uuid
    
    start_time = time.time()
    session_id = request.session_id or await voice_service.create_session(str(current_user.id))
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Check for wake word if enabled
        if request.session_id is None:  # New session, check wake word
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
        
        # Process through JARVIS
        context = {"session_id": session_id} if request.include_context else {}
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
        logger.error(f"Voice processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice processing failed: {str(e)}"
        )


@router.post("/transcribe")
async def transcribe_audio(
    current_user: Annotated[User, Depends(get_current_active_user)],
    audio_file: UploadFile = File(...),
    language: str = "en"
) -> Dict[str, Any]:
    """
    Transcribe audio file to text using multiple ASR providers
    """
    try:
        # Read audio file
        audio_content = await audio_file.read()
        
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
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/synthesize")
async def text_to_speech(
    request: TTSRequest,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict[str, Any]:
    """
    Convert text to speech and return audio data
    """
    try:
        # Use real voice service for TTS
        audio_bytes = await voice_service.synthesize_speech(
            text=request.text,
            voice=request.voice if request.voice != "default" else None,
            save_to_file=True  # Get bytes instead of direct playback
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
                "duration_estimate": len(request.text) * 0.06  # Rough estimate
            }
        else:
            # Fallback response if TTS not available
            return {
                "status": "partial",
                "text": request.text,
                "voice": request.voice,
                "format": request.format,
                "audio_data": None,
                "message": "TTS synthesis completed but audio generation failed"
            }
        
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text-to-speech failed: {str(e)}"
        )


@router.get("/voices")
async def list_voices() -> Dict[str, Any]:
    """
    List available TTS voices
    """
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        voice_list = []
        for voice in voices:
            voice_list.append({
                "id": voice.id,
                "name": voice.name,
                "languages": getattr(voice, 'languages', ['en-US']),
                "gender": getattr(voice, 'gender', 'unknown')
            })
        
        return {
            "voices": voice_list,
            "default": "default",
            "count": len(voice_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        return {
            "voices": [],
            "default": "default",
            "count": 0,
            "error": str(e)
        }


@router.get("/asr-providers")
async def list_asr_providers() -> Dict[str, Any]:
    """
    List available ASR (Automatic Speech Recognition) providers
    """
    providers = []
    
    # Check Whisper
    try:
        import whisper
        providers.append({
            "name": "whisper",
            "status": "available",
            "models": ["tiny", "base", "small", "medium", "large"],
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
            "models": ["small", "large"],
            "languages": "multiple"
        })
    except ImportError:
        providers.append({
            "name": "vosk",
            "status": "not_installed"
        })
    
    # Google Speech API is always available through speech_recognition
    providers.append({
        "name": "google",
        "status": "available",
        "requires_internet": True,
        "languages": "multiple"
    })
    
    return {
        "providers": providers,
        "default": "auto",
        "count": len([p for p in providers if p.get("status") == "available"])
    }


@router.get("/health")
async def voice_health_check() -> Dict[str, Any]:
    """
    Check voice processing system health
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "metrics": {}
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
async def voice_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice streaming
    Implements bidirectional audio streaming with JARVIS
    
    Protocol:
    Client -> Server: {"type": "audio", "data": base64_audio}
    Client -> Server: {"type": "control", "command": "start|stop|pause"}
    Server -> Client: {"type": "transcription", "text": "..."}
    Server -> Client: {"type": "response", "text": "...", "audio": base64_audio}
    Server -> Client: {"type": "status", "message": "..."}
    """
    await websocket.accept()
    
    session_id = None
    audio_buffer = []
    is_recording = False
    
    try:
        # Create voice session
        session_id = await voice_service.create_session()
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "message": "Connected to JARVIS voice stream",
            "session_id": session_id
        })
        
        # Start listening for messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_json()
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
                                
                                # Process through JARVIS
                                context = {"session_id": session_id}
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
                        # Process through JARVIS
                        context = {"session_id": session_id}
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
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up session
        if session_id:
            session = voice_service.get_session(session_id)
            if session:
                session.is_active = False


@router.post("/record")
async def record_voice(
    current_user: Annotated[User, Depends(get_current_active_user)],
    duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Record audio from microphone
    Returns base64 encoded audio data
    """
    try:
        # Record audio
        audio_bytes = await voice_service.record_audio(
            duration=duration,
            detect_silence=duration is None
        )
        
        if not audio_bytes:
            raise HTTPException(
                status_code=500,
                detail="Failed to record audio"
            )
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "status": "success",
            "audio_data": audio_base64,
            "duration": duration,
            "size": len(audio_bytes),
            "sample_rate": voice_service.config.rate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Recording failed: {str(e)}"
        )


@router.post("/wake-word/configure")
async def configure_wake_word(
    keywords: List[str],
    sensitivity: float = 0.5,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
) -> Dict[str, Any]:
    """
    Configure wake word detection settings
    """
    try:
        # Update wake word configuration
        wake_word_detector.set_keywords(keywords)
        wake_word_detector.set_sensitivity(sensitivity)
        
        return {
            "status": "success",
            "keywords": keywords,
            "sensitivity": sensitivity,
            "engine": wake_word_detector.config.engine.value
        }
        
    except Exception as e:
        logger.error(f"Wake word configuration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Configuration failed: {str(e)}"
        )


@router.post("/wake-word/test")
async def test_wake_word(
    audio_file: UploadFile = File(...),
    current_user: Annotated[User, Depends(get_current_active_user)] = None
) -> Dict[str, Any]:
    """
    Test wake word detection on an audio file
    """
    try:
        # Read audio file
        audio_content = await audio_file.read()
        
        # Test wake word detection
        detection = await wake_word_detector.detect(audio_content)
        
        return {
            "detected": detection.detected,
            "keyword": detection.keyword,
            "confidence": detection.confidence,
            "filename": audio_file.filename,
            "engine": wake_word_detector.config.engine.value
        }
        
    except Exception as e:
        logger.error(f"Wake word test error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        )


@router.get("/sessions")
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_active_user)] = None
) -> Dict[str, Any]:
    """
    List active voice sessions
    """
    try:
        active_sessions = [
            {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "is_active": session.is_active,
                "command_count": session.command_count,
                "wake_word_count": session.wake_word_count,
                "error_count": session.error_count
            }
            for session in voice_service.sessions.values()
            if session.is_active
        ]
        
        return {
            "sessions": active_sessions,
            "total_active": len(active_sessions),
            "metrics": voice_service.get_metrics()
        }
        
    except Exception as e:
        logger.error(f"Session list error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.post("/continuous-listening/start")
async def start_continuous_listening(
    current_user: Annotated[User, Depends(get_current_active_user)] = None
) -> Dict[str, Any]:
    """
    Start continuous listening mode with wake word detection
    """
    try:
        # Define callback for processing commands
        async def command_callback(text: str) -> str:
            # Process through JARVIS
            result = await jarvis.process(text, {})
            return result.get("response", "")
        
        # Start continuous listening in background
        import asyncio
        asyncio.create_task(
            voice_service.start_continuous_listening(callback=command_callback)
        )
        
        return {
            "status": "success",
            "message": "Continuous listening started",
            "wake_words": wake_word_detector.config.keywords
        }
        
    except Exception as e:
        logger.error(f"Continuous listening error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start continuous listening: {str(e)}"
        )


@router.post("/continuous-listening/stop")
async def stop_continuous_listening(
    current_user: Annotated[User, Depends(get_current_active_user)] = None
) -> Dict[str, Any]:
    """
    Stop continuous listening mode
    """
    try:
        voice_service.is_listening = False
        
        return {
            "status": "success",
            "message": "Continuous listening stopped",
            "metrics": voice_service.get_metrics()
        }
        
    except Exception as e:
        logger.error(f"Stop listening error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop listening: {str(e)}"
        )