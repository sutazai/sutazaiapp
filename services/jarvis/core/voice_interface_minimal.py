#!/usr/bin/env python3
"""
Minimal Voice Interface for JARVIS AI System
Basic voice capabilities without heavy ML dependencies
"""

import asyncio
import logging
import json
import os
import tempfile
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceInterface:
    """Minimal voice interface with basic speech capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recognizer = None
        self.tts_engine = None
        self.microphone = None
        
        # Voice activity detection settings
        self.sample_rate = config.get('sample_rate', 16000)
        
        # Wake word settings
        self.wake_words = config.get('wake_words', ['jarvis', 'hey jarvis', 'computer'])
        self.wake_word_threshold = config.get('wake_word_threshold', 0.7)
        
        # Speech recognition settings
        self.recognition_language = config.get('language', 'en-US')
        self.recognition_timeout = config.get('recognition_timeout', 5.0)
        
        # TTS settings
        self.tts_rate = config.get('tts_rate', 200)
        self.tts_volume = config.get('tts_volume', 0.9)
        
        # Callbacks
        self.command_callback: Optional[Callable] = None
        self.wake_word_callback: Optional[Callable] = None
        
        # State
        self.is_listening = False
        self.is_recording = False
        self.continuous_mode = False
        
    async def initialize(self):
        """Initialize voice components"""
        try:
            # Initialize speech recognition if available
            if SPEECH_RECOGNITION_AVAILABLE:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 4000
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                
                # Initialize microphone
                try:
                    self.microphone = sr.Microphone()
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    logger.info("Microphone initialized and calibrated")
                except Exception as e:
                    logger.warning(f"Microphone initialization failed: {e}")
                    self.microphone = None
            else:
                logger.warning("Speech recognition not available - install SpeechRecognition")
                
            # Initialize TTS engine if available
            if TTS_AVAILABLE:
                try:
                    self.tts_engine = pyttsx3.init()
                    self.tts_engine.setProperty('rate', self.tts_rate)
                    self.tts_engine.setProperty('volume', self.tts_volume)
                    
                    # Set voice if available
                    voices = self.tts_engine.getProperty('voices')
                    if voices:
                        for voice in voices:
                            if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                                self.tts_engine.setProperty('voice', voice.id)
                                break
                    logger.info("TTS engine initialized")
                except Exception as e:
                    logger.warning(f"TTS initialization failed: {e}")
                    self.tts_engine = None
            else:
                logger.warning("TTS not available - install pyttsx3")
                
            logger.info("Voice interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice interface: {e}")
            
    async def shutdown(self):
        """Shutdown voice interface"""
        try:
            self.is_listening = False
            self.is_recording = False
            
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
                    
            logger.info("Voice interface shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during voice interface shutdown: {e}")
            
    def set_command_callback(self, callback: Callable):
        """Set callback for voice commands"""
        self.command_callback = callback
        
    def set_wake_word_callback(self, callback: Callable):
        """Set callback for wake word detection"""
        self.wake_word_callback = callback
        
    async def start_continuous_listening(self):
        """Start continuous voice monitoring"""
        if not self.is_available():
            logger.warning("Voice interface not available - speech recognition disabled")
            return
            
        if self.is_listening:
            return
            
        self.is_listening = True
        self.continuous_mode = True
        
        logger.info("Starting continuous voice listening")
        
        # Start background listening task
        asyncio.create_task(self._continuous_listen_loop())
        
    async def stop_continuous_listening(self):
        """Stop continuous voice monitoring"""
        self.is_listening = False
        self.continuous_mode = False
        logger.info("Stopped continuous voice listening")
        
    async def _continuous_listen_loop(self):
        """Main loop for continuous voice processing"""
        while self.is_listening and self.microphone and self.recognizer:
            try:
                # Listen for audio
                with self.microphone as source:
                    # Short timeout for responsiveness
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1, 
                        phrase_time_limit=5
                    )
                    
                # Process the audio
                if audio:
                    await self._process_audio_chunk(audio)
                    
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                logger.error(f"Error in continuous listening: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
                
    async def _process_audio_chunk(self, audio):
        """Process audio chunk for wake words and commands"""
        try:
            # Quick recognition for wake word detection
            try:
                text = self.recognizer.recognize_google(
                    audio, 
                    language=self.recognition_language
                ).lower()
                
                # Check for wake words
                if self._detect_wake_word(text):
                    logger.info(f"Wake word detected: {text}")
                    
                    if self.wake_word_callback:
                        await self.wake_word_callback(text)
                        
                    # Listen for command after wake word
                    command = await self._listen_for_command()
                    if command and self.command_callback:
                        await self.command_callback(command)
                        
            except sr.UnknownValueError:
                # No speech detected, continue
                pass
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            
    def _detect_wake_word(self, text: str) -> bool:
        """Detect if wake word is present in text"""        
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if wake_word.lower() in text_lower:
                return True
                
        return False
        
    async def _listen_for_command(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for command after wake word detection"""
        try:
            if not self.microphone or not self.recognizer:
                return None
                
            logger.info("Listening for voice command...")
            
            with self.microphone as source:
                # Longer timeout for command
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=10
                )
                
            # Use Google Speech Recognition
            try:
                command = self.recognizer.recognize_google(
                    audio, 
                    language=self.recognition_language
                )
                logger.info(f"Command recognized: {command}")
                return command
                
            except sr.UnknownValueError:
                logger.info("Could not understand the command")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            logger.info("No command received within timeout")
            return None
        except Exception as e:
            logger.error(f"Error listening for command: {e}")
            return None
            
    async def speech_to_text(self, audio_path: str) -> Optional[str]:
        """Convert audio file to text"""
        try:
            if not self.recognizer:
                return None
                
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                
            # Use Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(
                    audio, 
                    language=self.recognition_language
                )
                return text.strip()
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error in speech to text: {e}")
            return None
            
    async def text_to_speech(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Convert text to speech"""
        try:
            if not text.strip() or not self.tts_engine:
                return None
                
            logger.info(f"Converting to speech: {text[:50]}...")
            
            if output_path:
                # Save to file
                self.tts_engine.save_to_file(text, output_path)
                self.tts_engine.runAndWait()
                return output_path
            else:
                # Speak directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None
                
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            return None
            
    def is_available(self) -> bool:
        """Check if voice interface is available"""
        return (
            SPEECH_RECOGNITION_AVAILABLE and 
            self.recognizer is not None and 
            self.microphone is not None
        )
        
    async def get_voice_metrics(self) -> Dict[str, Any]:
        """Get voice interface performance metrics"""
        return {
            'is_listening': self.is_listening,
            'is_recording': self.is_recording,
            'continuous_mode': self.continuous_mode,
            'recognition_language': self.recognition_language,
            'microphone_available': self.microphone is not None,
            'tts_available': self.tts_engine is not None,
            'speech_recognition_available': SPEECH_RECOGNITION_AVAILABLE,
            'pyttsx3_available': TTS_AVAILABLE
        }