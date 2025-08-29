"""Voice processing endpoints for JARVIS"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import io
import logging
from datetime import datetime

from app.services.voice_pipeline import VoicePipeline, VoiceConfig, ASRProvider, TTSProvider
from app.services.jarvis_orchestrator import JARVISOrchestrator
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Check if speech_recognition is available
try:
    if not SR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Speech recognition not available - dependencies not installed"
        )
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
    if not SR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Speech recognition not available - dependencies not installed"
        )
    
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        # Initialize voice pipeline for this request
        async def process_callback(text: str) -> str:
            """Process recognized text through JARVIS"""
            result = await jarvis.process(text, {"session_id": session_id})
            return result.get("response", "")
        
        pipeline = VoicePipeline(voice_config, process_callback)
        
        # Convert audio bytes to AudioData
        if SR_AVAILABLE:
            audio_data = sr.AudioData(audio_bytes, 16000, 2)
        else:
            raise HTTPException(
                status_code=503,
                detail="Speech recognition not available"
            )
        
        # Recognize speech
        recognized_text = await pipeline._recognize_speech(audio_data)
        
        if not recognized_text:
            raise HTTPException(
                status_code=400,
                detail="Could not recognize speech from audio"
            )
        
        # Process through JARVIS
        context = {"session_id": session_id} if request.include_context else {}
        jarvis_response = await jarvis.process(recognized_text, context)
        
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
    audio_file: UploadFile = File(...),
    language: str = "en-US"
) -> Dict[str, Any]:
    """
    Transcribe audio file to text using multiple ASR providers
    """
    try:
        # Read audio file
        audio_content = await audio_file.read()
        
        # Initialize voice pipeline
        pipeline = VoicePipeline(voice_config, None)
        
        # Convert to AudioData
        if not SR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Speech recognition not available - dependencies not installed"
            )
        audio_data = sr.AudioData(audio_content, 16000, 2)
        
        # Recognize speech
        text = await pipeline._recognize_speech(audio_data)
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio"
            )
        
        return {
            "text": text,
            "language": language,
            "filename": audio_file.filename,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/synthesize")
async def text_to_speech(request: TTSRequest) -> Dict[str, Any]:
    """
    Convert text to speech and return audio data
    """
    try:
        # Initialize voice pipeline
        pipeline = VoicePipeline(voice_config, None)
        
        # Generate speech (this is a simplified version)
        # In production, this would generate actual audio data
        import pyttsx3
        import io
        import wave
        
        # For now, return a placeholder response
        # Real implementation would generate audio
        
        return {
            "status": "success",
            "text": request.text,
            "voice": request.voice,
            "format": request.format,
            "audio_data": None,  # Would contain base64 encoded audio
            "message": "TTS endpoint configured but audio generation not fully implemented"
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
        "components": {}
    }
    
    # Check ASR
    try:
        if not SR_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Speech recognition not available - dependencies not installed"
            )
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