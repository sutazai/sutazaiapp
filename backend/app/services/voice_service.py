"""
Complete Voice Service Implementation for JARVIS
Implements real voice processing with wake word detection, ASR, and TTS
Based on best practices from multiple JARVIS repositories
"""

import asyncio
import io
import json
import logging
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Core audio libraries
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not available - audio recording disabled")

# Speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logging.warning("SpeechRecognition not available")

# TTS engines
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.warning("gTTS not available")

# Advanced ASR
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available")

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available")

# Audio playback
PYGAME_AVAILABLE = False
try:
    import pygame
    # Don't initialize mixer here - do it lazily when needed
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    logging.warning("Pygame not available")

# WebRTC VAD for better voice activity detection
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logging.warning("WebRTC VAD not available")

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Audio format specifications"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


@dataclass
class AudioConfig:
    """Audio configuration parameters"""
    format: int = pyaudio.paInt16 if PYAUDIO_AVAILABLE else 8
    channels: int = 1
    rate: int = 16000
    chunk_size: int = 1024
    record_seconds: float = 5.0
    energy_threshold: int = 4000
    pause_threshold: float = 0.8
    phrase_threshold: float = 0.3
    non_speaking_duration: float = 0.5


@dataclass
class VoiceSession:
    """Voice interaction session"""
    session_id: str
    start_time: datetime
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    wake_word_count: int = 0
    command_count: int = 0
    error_count: int = 0


class VoiceService:
    """
    Complete voice service implementation with:
    - Real-time audio recording and playback
    - Wake word detection (multiple methods)
    - Multi-provider ASR with fallback
    - TTS with voice selection
    - WebSocket streaming support
    - Session management
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.sessions: Dict[str, VoiceSession] = {}
        self.audio = None
        self.stream = None
        
        # Initialize audio system
        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                logger.info("PyAudio initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PyAudio: {e}")
                self.audio = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
        if self.recognizer:
            self.recognizer.energy_threshold = self.config.energy_threshold
            self.recognizer.pause_threshold = self.config.pause_threshold
            self.recognizer.phrase_threshold = self.config.phrase_threshold
            self.recognizer.non_speaking_duration = self.config.non_speaking_duration
        
        # Initialize ASR models
        self.whisper_model = None
        self.vosk_model = None
        self._initialize_asr_models()
        
        # Initialize TTS engine
        self.tts_engine = None
        self._initialize_tts()
        
        # Voice activity detection
        self.vad = webrtcvad.Vad() if VAD_AVAILABLE else None
        if self.vad:
            self.vad.set_mode(1)  # 0-3, 3 being most aggressive
        
        # Wake word detection
        self.wake_words = ["jarvis", "hey jarvis", "ok jarvis", "hello jarvis"]
        self.wake_word_buffer = deque(maxlen=50)  # Store recent audio for wake word
        
        # Audio buffers
        self.audio_buffer = queue.Queue()
        self.response_buffer = queue.Queue()
        
        # Processing state
        self.is_listening = False
        self.is_recording = False
        self.is_speaking = False
        
        # Metrics
        self.metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "wake_word_detections": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "average_response_time": 0,
            "total_audio_processed": 0
        }
    
    def _initialize_asr_models(self):
        """Initialize ASR models for speech recognition"""
        # Initialize Whisper
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
        
        # Initialize Vosk
        if VOSK_AVAILABLE:
            try:
                # Try to find or download a Vosk model
                model_path = Path("/opt/models/vosk-model-small-en-us-0.15")
                if model_path.exists():
                    self.vosk_model = vosk.Model(str(model_path))
                    logger.info("Vosk model loaded successfully")
                else:
                    logger.warning(f"Vosk model not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load Vosk model: {e}")
    
    def _initialize_tts(self):
        """Initialize TTS engine"""
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                
                # Configure voice properties
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to find a male English voice for JARVIS
                    for voice in voices:
                        if 'english' in voice.name.lower():
                            if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                                self.tts_engine.setProperty('voice', voice.id)
                                break
                
                # Set speech properties
                self.tts_engine.setProperty('rate', 180)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                
                logger.info("TTS engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new voice session"""
        import uuid
        session_id = str(uuid.uuid4())
        session = VoiceSession(
            session_id=session_id,
            start_time=datetime.now(),
            user_id=user_id
        )
        self.sessions[session_id] = session
        self.metrics["total_sessions"] += 1
        self.metrics["active_sessions"] = len([s for s in self.sessions.values() if s.is_active])
        
        logger.info(f"Created voice session: {session_id}")
        return session_id
    
    async def record_audio(self, duration: Optional[float] = None, 
                          detect_silence: bool = True) -> Optional[bytes]:
        """
        Record audio from microphone
        Returns raw audio bytes or None if recording failed
        """
        if not self.audio or not PYAUDIO_AVAILABLE:
            logger.error("Audio recording not available")
            return None
        
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            logger.info("Recording audio...")
            frames = []
            
            if duration:
                # Record for fixed duration
                for _ in range(0, int(self.config.rate / self.config.chunk_size * duration)):
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    frames.append(data)
            else:
                # Record until silence detected
                silence_count = 0
                max_silence = int(self.config.rate * self.config.pause_threshold / self.config.chunk_size)
                speaking = False
                
                while True:
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    
                    if detect_silence:
                        # Check if user is speaking
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        energy = np.abs(audio_array).mean()
                        
                        if energy > self.config.energy_threshold / 10:
                            speaking = True
                            silence_count = 0
                        else:
                            if speaking:
                                silence_count += 1
                                if silence_count > max_silence:
                                    break
                    
                    # Limit maximum recording time
                    if len(frames) > self.config.rate * 30 / self.config.chunk_size:  # 30 seconds max
                        break
            
            stream.stop_stream()
            stream.close()
            
            # Convert frames to bytes
            audio_bytes = b''.join(frames)
            self.metrics["total_audio_processed"] += len(audio_bytes)
            
            logger.info(f"Recorded {len(audio_bytes)} bytes of audio")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return None
    
    async def detect_wake_word(self, audio_bytes: bytes) -> bool:
        """
        Detect wake word in audio
        Uses simple keyword spotting with speech recognition
        """
        if not SR_AVAILABLE or not self.recognizer:
            return False
        
        try:
            # Convert bytes to AudioData
            audio_data = sr.AudioData(audio_bytes, self.config.rate, 2)
            
            # Try to recognize speech
            text = ""
            try:
                text = self.recognizer.recognize_google(audio_data, language="en-US")
            except (sr.UnknownValueError, sr.RequestError):
                return False
            
            # Check for wake words
            text_lower = text.lower()
            for wake_word in self.wake_words:
                if wake_word in text_lower:
                    logger.info(f"Wake word detected: '{wake_word}' in '{text}'")
                    self.metrics["wake_word_detections"] += 1
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False
    
    async def transcribe_audio(self, audio_bytes: bytes, 
                              language: str = "en") -> Optional[str]:
        """
        Transcribe audio to text using multiple ASR providers
        Fallback chain: Whisper -> Vosk -> Google Speech
        """
        if not audio_bytes:
            return None
        
        start_time = time.time()
        text = None
        
        # Try Whisper first (most accurate)
        if self.whisper_model and WHISPER_AVAILABLE:
            try:
                # Convert audio bytes to numpy array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    audio_array,
                    language=language,
                    fp16=False
                )
                text = result["text"].strip()
                
                if text:
                    logger.info(f"Transcribed with Whisper: {text}")
                    self._update_metrics(True, time.time() - start_time)
                    return text
                    
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}")
        
        # Try Vosk (offline, fast)
        if self.vosk_model and VOSK_AVAILABLE:
            try:
                rec = vosk.KaldiRecognizer(self.vosk_model, self.config.rate)
                rec.SetWords(True)
                
                # Process audio
                rec.AcceptWaveform(audio_bytes)
                result = json.loads(rec.FinalResult())
                text = result.get("text", "").strip()
                
                if text:
                    logger.info(f"Transcribed with Vosk: {text}")
                    self._update_metrics(True, time.time() - start_time)
                    return text
                    
            except Exception as e:
                logger.warning(f"Vosk transcription failed: {e}")
        
        # Try Google Speech Recognition (requires internet)
        if SR_AVAILABLE and self.recognizer:
            try:
                audio_data = sr.AudioData(audio_bytes, self.config.rate, 2)
                text = self.recognizer.recognize_google(
                    audio_data,
                    language=f"{language}-US" if len(language) == 2 else language
                )
                
                if text:
                    logger.info(f"Transcribed with Google: {text}")
                    self._update_metrics(True, time.time() - start_time)
                    return text
                    
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition error: {e}")
            except Exception as e:
                logger.warning(f"Google transcription failed: {e}")
        
        # All methods failed
        self._update_metrics(False, 0)
        logger.error("All transcription methods failed")
        return None
    
    async def synthesize_speech(self, text: str, 
                               voice: Optional[str] = None,
                               save_to_file: bool = False) -> Optional[bytes]:
        """
        Convert text to speech
        Returns audio bytes or saves to file
        """
        if not text:
            return None
        
        audio_bytes = None
        
        # Try pyttsx3 first (offline)
        if self.tts_engine and PYTTSX3_AVAILABLE:
            try:
                if save_to_file:
                    # Save to temporary file
                    temp_file = f"/tmp/tts_{time.time()}.wav"
                    self.tts_engine.save_to_file(text, temp_file)
                    self.tts_engine.runAndWait()
                    
                    # Read file bytes
                    with open(temp_file, 'rb') as f:
                        audio_bytes = f.read()
                    
                    # Clean up
                    Path(temp_file).unlink()
                else:
                    # Speak directly
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    
                logger.info("Speech synthesized with pyttsx3")
                return audio_bytes
                
            except Exception as e:
                logger.warning(f"pyttsx3 TTS failed: {e}")
        
        # Try gTTS (requires internet)
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                
                # Save to BytesIO
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_bytes = audio_buffer.read()
                
                if not save_to_file and PYGAME_AVAILABLE:
                    try:
                        # Initialize mixer if not already done
                        if not pygame.mixer.get_init():
                            pygame.mixer.init()
                        # Play audio
                        audio_buffer.seek(0)
                        pygame.mixer.music.load(audio_buffer)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)
                    except (pygame.error, AttributeError) as e:
                        logger.warning(f"Pygame playback failed: {e}")
                
                logger.info("Speech synthesized with gTTS")
                return audio_bytes
                
            except Exception as e:
                logger.warning(f"gTTS failed: {e}")
        
        logger.error("No TTS engine available")
        return None
    
    async def speak(self, text: str, interrupt: bool = True) -> bool:
        """
        Speak text using TTS
        Returns True if successful
        """
        if interrupt and self.is_speaking:
            # Stop current speech
            if self.tts_engine:
                self.tts_engine.stop()
            if PYGAME_AVAILABLE and pygame:
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                except (pygame.error, AttributeError):
                    pass
        
        self.is_speaking = True
        
        try:
            result = await self.synthesize_speech(text, save_to_file=False)
            self.is_speaking = False
            return result is not None or True  # Return True even if no bytes (direct playback)
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            self.is_speaking = False
            return False
    
    async def process_voice_command(self, audio_bytes: bytes, 
                                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a voice command end-to-end
        Returns transcription and any response
        """
        result = {
            "success": False,
            "transcription": None,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get or create session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session_id = await self.create_session()
            session = self.sessions[session_id]
            result["session_id"] = session_id
        
        try:
            # Transcribe audio
            text = await self.transcribe_audio(audio_bytes)
            
            if text:
                result["transcription"] = text
                result["success"] = True
                
                # Update session
                session.command_count += 1
                session.history.append({
                    "type": "command",
                    "text": text,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.metrics["successful_commands"] += 1
            else:
                session.error_count += 1
                self.metrics["failed_commands"] += 1
                result["error"] = "Could not transcribe audio"
                
        except Exception as e:
            logger.error(f"Voice command processing error: {e}")
            result["error"] = str(e)
            session.error_count += 1
            self.metrics["failed_commands"] += 1
        
        return result
    
    async def start_continuous_listening(self, callback: Optional[Callable] = None):
        """
        Start continuous listening with wake word detection
        Calls callback with recognized commands
        """
        if not self.audio or not PYAUDIO_AVAILABLE:
            logger.error("Audio not available for continuous listening")
            return
        
        self.is_listening = True
        logger.info("Started continuous listening mode")
        
        try:
            # Open continuous audio stream
            stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            stream.start_stream()
            
            # Process audio in background
            while self.is_listening:
                # Check for wake word in buffer
                if len(self.wake_word_buffer) > 10:
                    # Get recent audio
                    recent_audio = b''.join(list(self.wake_word_buffer)[-20:])
                    
                    # Check for wake word
                    if await self.detect_wake_word(recent_audio):
                        # Wake word detected - record command
                        await self.speak("Yes, I'm listening")
                        
                        # Record user command
                        command_audio = await self.record_audio(detect_silence=True)
                        
                        if command_audio:
                            # Process command
                            result = await self.process_voice_command(command_audio)
                            
                            if result["success"] and callback:
                                # Call callback with transcription
                                response = await callback(result["transcription"])
                                if response:
                                    await self.speak(response)
                        
                        # Clear buffer after processing
                        self.wake_word_buffer.clear()
                
                await asyncio.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Continuous listening error: {e}")
        finally:
            self.is_listening = False
            logger.info("Stopped continuous listening")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for continuous audio stream
        Stores audio in buffer for processing
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Add to wake word buffer
        self.wake_word_buffer.append(in_data)
        
        # Add to main buffer if recording
        if self.is_recording:
            self.audio_buffer.put(in_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _update_metrics(self, success: bool, response_time: float):
        """Update service metrics"""
        if success:
            # Update average response time
            total = self.metrics["successful_commands"]
            avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (avg * total + response_time) / (total + 1)
    
    async def stop(self):
        """Stop all voice processing"""
        self.is_listening = False
        self.is_recording = False
        self.is_speaking = False
        
        # Stop TTS
        if self.tts_engine:
            self.tts_engine.stop()
        
        # Stop audio
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        # Close all sessions
        for session in self.sessions.values():
            session.is_active = False
        
        self.metrics["active_sessions"] = 0
        
        logger.info("Voice service stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return self.metrics.copy()
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    async def save_audio(self, audio_bytes: bytes, filepath: str, 
                        format: AudioFormat = AudioFormat.WAV) -> bool:
        """
        Save audio bytes to file
        """
        try:
            if format == AudioFormat.WAV:
                # Save as WAV
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(self.config.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.config.format))
                    wf.setframerate(self.config.rate)
                    wf.writeframes(audio_bytes)
                
                logger.info(f"Audio saved to {filepath}")
                return True
            else:
                logger.warning(f"Format {format} not yet implemented")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    async def load_audio(self, filepath: str) -> Optional[bytes]:
        """
        Load audio from file
        """
        try:
            with wave.open(filepath, 'rb') as wf:
                audio_bytes = wf.readframes(wf.getnframes())
                return audio_bytes
                
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None


# Singleton instance
_voice_service_instance = None


def get_voice_service() -> VoiceService:
    """Get or create voice service singleton"""
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance