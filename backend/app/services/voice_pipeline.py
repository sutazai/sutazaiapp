"""
Advanced Voice Pipeline for JARVIS
Integrates Whisper ASR, Vosk, Google Speech API, and wake word detection
Based on best practices from Dipeshpal/Jarvis_AI and JARVIS-AGI
"""

import asyncio
import io
import logging
import queue
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Speech recognition providers with fallback
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logging.warning("SpeechRecognition not available")

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

import json

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not available")

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

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available")

# Wake word detection
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    logging.warning("Porcupine not available for wake word detection")

logger = logging.getLogger(__name__)


class ASRProvider(Enum):
    """Available ASR providers"""
    WHISPER = "whisper"
    VOSK = "vosk"
    GOOGLE = "google"
    AUTO = "auto"  # Automatically select best available


class TTSProvider(Enum):
    """Available TTS providers"""
    PYTTSX3 = "pyttsx3"
    GOOGLE_TTS = "google_tts"
    ELEVENLABS = "elevenlabs"
    AUTO = "auto"


@dataclass
class VoiceConfig:
    """Voice pipeline configuration"""
    wake_word: str = "jarvis"
    asr_provider: ASRProvider = ASRProvider.AUTO
    tts_provider: TTSProvider = TTSProvider.AUTO
    language: str = "en-US"
    energy_threshold: int = 4000
    pause_threshold: float = 0.8
    whisper_model: str = "base"
    vosk_model_path: Optional[str] = None
    enable_wake_word: bool = True
    enable_interruption: bool = True
    audio_format: int = 8 if not PYAUDIO_AVAILABLE else pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk_size: int = 512


class VoicePipeline:
    """
    Advanced voice processing pipeline combining best features from:
    - Dipeshpal/Jarvis_AI: Server-based processing, extensible
    - JARVIS-AGI: Vosk integration, interruption handling
    - Microsoft JARVIS: Pipeline architecture
    """
    
    def __init__(self, config: VoiceConfig, on_command_callback: Optional[Callable] = None):
        self.config = config
        self.on_command_callback = on_command_callback
        
        # Initialize components
        self.audio = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
        if self.recognizer:
            self.recognizer.energy_threshold = config.energy_threshold
            self.recognizer.pause_threshold = config.pause_threshold
        
        # Initialize ASR providers
        self.whisper_model = None
        self.vosk_model = None
        self.google_recognizer = sr.Recognizer()
        self._initialize_asr()
        
        # Initialize TTS
        self.tts_engine = None
        self._initialize_tts()
        
        # Initialize wake word detection
        self.porcupine = None
        if config.enable_wake_word and PORCUPINE_AVAILABLE:
            self._initialize_wake_word()
        
        # State management
        self.is_listening = False
        self.is_processing = False
        self.should_stop = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Metrics
        self.metrics = {
            "wake_word_detections": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "average_recognition_time": 0
        }
    
    def _initialize_asr(self):
        """Initialize ASR providers based on configuration"""
        if self.config.asr_provider in [ASRProvider.WHISPER, ASRProvider.AUTO]:
            try:
                self.whisper_model = whisper.load_model(self.config.whisper_model)
                logger.info(f"Whisper model '{self.config.whisper_model}' loaded")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
        
        if self.config.asr_provider in [ASRProvider.VOSK, ASRProvider.AUTO]:
            if self.config.vosk_model_path:
                try:
                    self.vosk_model = vosk.Model(self.config.vosk_model_path)
                    logger.info("Vosk model loaded")
                except Exception as e:
                    logger.error(f"Failed to load Vosk model: {e}")
    
    def _initialize_tts(self):
        """Initialize TTS engine"""
        if self.config.tts_provider in [TTSProvider.PYTTSX3, TTSProvider.AUTO]:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure voice properties
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Select a voice (prefer English male voice for JARVIS)
                    for voice in voices:
                        if 'english' in voice.name.lower() and 'male' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                self.tts_engine.setProperty('rate', 180)  # Speech rate
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                logger.info("pyttsx3 TTS engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
    
    def _initialize_wake_word(self):
        """Initialize Porcupine wake word detection"""
        try:
            # Initialize Porcupine with "Jarvis" wake word
            # Note: Requires Porcupine access key
            self.porcupine = pvporcupine.create(
                keywords=[self.config.wake_word],
                sensitivities=[0.5]
            )
            logger.info(f"Wake word detection initialized for '{self.config.wake_word}'")
        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            self.porcupine = None
    
    async def start_listening(self):
        """
        Start the voice pipeline in listening mode
        Implements continuous listening with wake word detection
        """
        self.is_listening = True
        self.should_stop = False
        
        # Start audio stream
        stream = self.audio.open(
            format=self.config.audio_format,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk_size
        )
        
        logger.info("Voice pipeline started, listening...")
        
        try:
            while self.is_listening and not self.should_stop:
                if self.config.enable_wake_word and self.porcupine:
                    # Wait for wake word
                    if await self._detect_wake_word(stream):
                        self.metrics["wake_word_detections"] += 1
                        await self._process_command(stream)
                else:
                    # Continuous listening without wake word
                    await self._process_command(stream)
                
                await asyncio.sleep(0.1)
        
        finally:
            stream.stop_stream()
            stream.close()
            logger.info("Voice pipeline stopped")
    
    async def _detect_wake_word(self, stream) -> bool:
        """
        Detect wake word using Porcupine
        Returns True if wake word detected
        """
        if not self.porcupine:
            return True  # Skip wake word detection if not available
        
        try:
            audio_frame = stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            audio_frame = np.frombuffer(audio_frame, dtype=np.int16)
            
            keyword_index = self.porcupine.process(audio_frame)
            if keyword_index >= 0:
                logger.info(f"Wake word '{self.config.wake_word}' detected!")
                await self.speak("Yes, I'm listening.")
                return True
        
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
        
        return False
    
    async def _process_command(self, stream):
        """
        Process voice command after wake word detection
        Implements multi-provider ASR with fallback
        """
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Record audio until pause
            logger.info("Recording audio...")
            audio_data = await self._record_audio(stream)
            
            if audio_data:
                # Try ASR providers in order of preference
                text = await self._recognize_speech(audio_data)
                
                if text:
                    recognition_time = time.time() - start_time
                    self._update_recognition_metrics(True, recognition_time)
                    logger.info(f"Recognized: {text}")
                    
                    # Process command through callback
                    if self.on_command_callback:
                        response = await self.on_command_callback(text)
                        if response:
                            await self.speak(response)
                else:
                    self._update_recognition_metrics(False, 0)
                    await self.speak("I didn't catch that. Could you please repeat?")
        
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            self._update_recognition_metrics(False, 0)
        
        finally:
            self.is_processing = False
    
    async def _record_audio(self, stream) -> Optional[sr.AudioData]:
        """
        Record audio until pause detected
        Implements interruption handling from JARVIS-AGI
        """
        frames = []
        silence_count = 0
        max_silence = int(self.config.rate * self.config.pause_threshold / self.config.chunk_size)
        
        while silence_count < max_silence:
            if self.should_stop or (self.config.enable_interruption and self.response_queue.qsize() > 0):
                break
            
            try:
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Simple silence detection
                audio_array = np.frombuffer(data, dtype=np.int16)
                if np.abs(audio_array).mean() < self.config.energy_threshold / 10:
                    silence_count += 1
                else:
                    silence_count = 0
            
            except Exception as e:
                logger.error(f"Audio recording error: {e}")
                break
        
        if frames:
            # Convert to AudioData format
            audio_bytes = b''.join(frames)
            return sr.AudioData(audio_bytes, self.config.rate, 2)
        
        return None
    
    async def _recognize_speech(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        Recognize speech using multiple providers with fallback
        Priority: Whisper -> Vosk -> Google
        """
        text = None
        
        # Try Whisper (most accurate, but slower)
        if self.whisper_model and self.config.asr_provider in [ASRProvider.WHISPER, ASRProvider.AUTO]:
            try:
                # Convert audio to format Whisper expects
                audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.whisper_model.transcribe(audio_array, language=self.config.language[:2])
                text = result["text"].strip()
                if text:
                    logger.info("Speech recognized with Whisper")
                    return text
            except Exception as e:
                logger.warning(f"Whisper ASR failed: {e}")
        
        # Try Vosk (fast, offline)
        if self.vosk_model and self.config.asr_provider in [ASRProvider.VOSK, ASRProvider.AUTO]:
            try:
                rec = vosk.KaldiRecognizer(self.vosk_model, self.config.rate)
                rec.AcceptWaveform(audio_data.frame_data)
                result = json.loads(rec.FinalResult())
                text = result.get("text", "").strip()
                if text:
                    logger.info("Speech recognized with Vosk")
                    return text
            except Exception as e:
                logger.warning(f"Vosk ASR failed: {e}")
        
        # Try Google Speech API (requires internet)
        if self.config.asr_provider in [ASRProvider.GOOGLE, ASRProvider.AUTO]:
            try:
                text = self.google_recognizer.recognize_google(
                    audio_data,
                    language=self.config.language
                )
                if text:
                    logger.info("Speech recognized with Google")
                    return text
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition error: {e}")
        
        return None
    
    async def speak(self, text: str, interrupt: bool = True):
        """
        Convert text to speech and play it
        Implements interruption capability from JARVIS-AGI
        """
        if interrupt and self.is_processing:
            # Add to response queue for interruption
            self.response_queue.put(text)
            return
        
        try:
            if self.config.tts_provider == TTSProvider.GOOGLE_TTS:
                # Use Google TTS
                tts = gTTS(text=text, lang=self.config.language[:2])
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                # Play using pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
            
            elif self.tts_engine:
                # Use pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            else:
                logger.warning("No TTS engine available")
        
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def _update_recognition_metrics(self, success: bool, recognition_time: float):
        """Update recognition metrics"""
        if success:
            self.metrics["successful_recognitions"] += 1
            n = self.metrics["successful_recognitions"]
            old_avg = self.metrics["average_recognition_time"]
            self.metrics["average_recognition_time"] = (old_avg * (n - 1) + recognition_time) / n
        else:
            self.metrics["failed_recognitions"] += 1
    
    def stop(self):
        """Stop the voice pipeline"""
        self.should_stop = True
        self.is_listening = False
        
        if self.porcupine:
            self.porcupine.delete()
        
        if self.tts_engine:
            self.tts_engine.stop()
        
        self.audio.terminate()
        logger.info("Voice pipeline terminated")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.metrics.copy()
    
    # Advanced features from repositories
    
    async def process_with_context(self, audio_data: sr.AudioData, context: Dict[str, Any]) -> Optional[str]:
        """
        Process audio with conversation context
        Inspired by JARVIS-AGI's session management
        """
        text = await self._recognize_speech(audio_data)
        
        if text and context:
            # Apply context to improve recognition
            # This could involve using previous conversation to improve accuracy
            pass
        
        return text
    
    async def batch_process_audio(self, audio_files: list) -> list:
        """
        Batch process multiple audio files
        Inspired by Dipeshpal's server architecture
        """
        results = []
        for audio_file in audio_files:
            # Load and process each file
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                text = await self._recognize_speech(audio_data)
                results.append(text)
        
        return results