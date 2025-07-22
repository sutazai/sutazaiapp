from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from pydantic import BaseModel
import os
import json
import asyncio
import base64
import wave
import io
from typing import List, Dict, Any, Optional
import whisper
import torch
import numpy as np
import logging
from datetime import datetime

app = FastAPI(title="RealtimeSTT Service", version="1.0.0")

class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    task: str = "transcribe"  # transcribe or translate
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    segments: List[Dict[str, Any]]
    processing_time: float

class STTManager:
    def __init__(self):
        self.model = None
        self.model_name = "base"  # Default model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        self.active_connections = set()
        
    def load_model(self, model_name: str = None):
        """Load Whisper model"""
        if model_name and model_name in self.supported_models:
            self.model_name = model_name
            
        if self.model is None or model_name:
            try:
                logging.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name, device=self.device)
                logging.info(f"Successfully loaded model {self.model_name} on {self.device}")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
                
    def process_audio_file(self, audio_data: bytes, request: TranscriptionRequest) -> TranscriptionResponse:
        """Process uploaded audio file"""
        import time
        start_time = time.time()
        
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
            
        try:
            # Save audio data to temporary file
            temp_audio_path = "/tmp/temp_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
                
            # Transcribe with Whisper
            result = self.model.transcribe(
                temp_audio_path,
                language=request.language,
                task=request.task,
                temperature=request.temperature,
                best_of=request.best_of,
                beam_size=request.beam_size,
                verbose=True
            )
            
            # Clean up temp file
            os.remove(temp_audio_path)
            
            processing_time = time.time() - start_time
            
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                segments=result["segments"],
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
            
    def process_audio_chunk(self, audio_chunk: bytes) -> str:
        """Process real-time audio chunk"""
        if self.model is None:
            self.load_model()
            
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Use Whisper for transcription
            result = self.model.transcribe(audio_np, fp16=False)
            return result["text"]
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {str(e)}")
            return ""

stt_manager = STTManager()

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "service": "realtimestt",
        "model_loaded": stt_manager.model is not None,
        "current_model": stt_manager.model_name
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    task: str = "transcribe",
    temperature: float = 0.0,
    best_of: int = 5,
    beam_size: int = 5
):
    """Transcribe uploaded audio file"""
    try:
        audio_data = await audio.read()
        
        request = TranscriptionRequest(
            language=language,
            task=task,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size
        )
        
        return stt_manager.process_audio_file(audio_data, request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription"""
    await websocket.accept()
    stt_manager.active_connections.add(websocket)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process audio chunk
            transcription = stt_manager.process_audio_chunk(data)
            
            # Send transcription back
            if transcription.strip():
                await websocket.send_json({
                    "transcription": transcription,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 1.0  # Whisper doesn't provide confidence scores directly
                })
                
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
    finally:
        stt_manager.active_connections.discard(websocket)

@app.get("/models")
def list_models():
    return {
        "available_models": stt_manager.supported_models,
        "current_model": stt_manager.model_name,
        "device": stt_manager.device
    }

@app.post("/models/{model_name}/load")
def load_model(model_name: str):
    if model_name not in stt_manager.supported_models:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
        
    try:
        stt_manager.load_model(model_name)
        return {"status": "loaded", "model": model_name, "device": stt_manager.device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    return {
        "model_loaded": stt_manager.model is not None,
        "current_model": stt_manager.model_name,
        "device": stt_manager.device,
        "active_connections": len(stt_manager.active_connections),
        "supported_models": stt_manager.supported_models,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/test")
def test_transcription():
    """Test endpoint with sample audio"""
    return {
        "message": "STT service is running",
        "upload_endpoint": "/transcribe",
        "websocket_endpoint": "/ws/realtime",
        "supported_formats": ["wav", "mp3", "m4a", "flac"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)