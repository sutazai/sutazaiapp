#!/usr/bin/env python3
"""
Advanced Voice Interface for JARVIS AI System
Provides sophisticated speech recognition, synthesis, and voice command processing
"""

import asyncio
import logging
import json
import os
import tempfile
import wave
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
import pyttsx3
import webrtcvad
import pyaudio
import numpy as np
from scipy import signal
from transformers import pipeline
import torch
import soundfile as sf

logger = logging.getLogger(__name__)

class VoiceInterface:
    """Advanced voice interface with real-time processing capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recognizer = None
        self.tts_engine = None
        self.microphone = None
        self.vad = None
        self.whisper_pipeline = None
        self.wake_word_detector = None
        self.noise_reducer = None
        
        # Voice activity detection settings
        self.vad_aggressiveness = config.get('vad_aggressiveness', 2)
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_duration = config.get('frame_duration', 30)  # ms
        
        # Wake word settings
        self.wake_words = config.get('wake_words', ['jarvis', 'hey jarvis', 'computer'])
        self.wake_word_threshold = config.get('wake_word_threshold', 0.7)
        
        # Speech recognition settings
        self.recognition_language = config.get('language', 'en-US')
        self.recognition_timeout = config.get('recognition_timeout', 5.0)
        self.phrase_timeout = config.get('phrase_timeout', 1.0)
        
        # TTS settings
        self.tts_rate = config.get('tts_rate', 200)
        self.tts_volume = config.get('tts_volume', 0.9)
        self.tts_voice = config.get('tts_voice', 'en-US-AriaNeural')
        
        # Callbacks
        self.command_callback: Optional[Callable] = None
        self.wake_word_callback: Optional[Callable] = None
        
        # State
        self.is_listening = False
        self.is_recording = False
        self.audio_buffer = []
        self.continuous_mode = False
        
    async def initialize(self):
        """Initialize all voice components"""
        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.dynamic_energy_adjustment_damping = 0.15
            self.recognizer.dynamic_energy_ratio = 1.5
            self.recognizer.pause_threshold = 0.8
            self.recognizer.operation_timeout = None
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.8
            
            # Initialize microphone
            try:
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone initialized and calibrated")
            except Exception as e:
                logger.warning(f"Microphone initialization failed: {e}")
                
            # Initialize TTS engine
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
                        
            # Initialize VAD
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            
            # Initialize Whisper for high-quality speech recognition
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
            try:
                self.whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en",
                    device=device
                )
                logger.info(f"Whisper pipeline initialized on {device}")
            except Exception as e:
                logger.warning(f"Whisper initialization failed: {e}")
                
            # Initialize wake word detection
            await self._initialize_wake_word_detection()
            
            logger.info("Voice interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice interface: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown voice interface"""
        try:
            self.is_listening = False
            self.is_recording = False
            
            if self.tts_engine:
                self.tts_engine.stop()
                
            logger.info("Voice interface shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during voice interface shutdown: {e}")
            
    async def _initialize_wake_word_detection(self):
        """Initialize wake word detection system"""
        try:
            # Simple wake word detection using keyword matching
            # In production, consider using specialized wake word models
            self.wake_word_detector = {
                'enabled': self.config.get('wake_word_enabled', True),
                'words': self.wake_words,
                'threshold': self.wake_word_threshold
            }
            logger.info("Wake word detection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            
    def set_command_callback(self, callback: Callable):
        """Set callback for voice commands"""
        self.command_callback = callback
        
    def set_wake_word_callback(self, callback: Callable):
        """Set callback for wake word detection"""
        self.wake_word_callback = callback
        
    async def start_continuous_listening(self):
        """Start continuous voice monitoring"""
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
        while self.is_listening and self.microphone:
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
        if not self.wake_word_detector['enabled']:
            return False
            
        text_lower = text.lower()
        for wake_word in self.wake_word_detector['words']:
            if wake_word.lower() in text_lower:
                return True
                
        return False
        
    async def _listen_for_command(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for command after wake word detection"""
        try:
            if not self.microphone:
                return None
                
            logger.info("Listening for voice command...")
            
            with self.microphone as source:
                # Longer timeout for command
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=10
                )
                
            # Use Whisper for better accuracy if available
            if self.whisper_pipeline and audio:
                try:
                    # Convert audio to numpy array
                    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Use Whisper for transcription
                    result = self.whisper_pipeline(audio_data)
                    command = result['text'].strip()
                    
                    logger.info(f"Command recognized (Whisper): {command}")
                    return command
                    
                except Exception as e:
                    logger.warning(f"Whisper recognition failed: {e}")
                    
            # Fallback to Google Speech Recognition
            try:
                command = self.recognizer.recognize_google(
                    audio, 
                    language=self.recognition_language
                )
                logger.info(f"Command recognized (Google): {command}")
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
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                
            # Try Whisper first if available
            if self.whisper_pipeline:
                try:
                    result = self.whisper_pipeline(audio_path)
                    return result['text'].strip()
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
                    
            # Fallback to Google Speech Recognition
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
            if not text.strip():
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
            
    async def record_audio(self, duration: float = 5.0, output_path: Optional[str] = None) -> Optional[str]:
        """Record audio for specified duration"""
        try:
            if not self.microphone:
                logger.error("No microphone available")
                return None
                
            if not output_path:
                output_path = f"/tmp/jarvis_recording_{datetime.now().timestamp()}.wav"
                
            logger.info(f"Recording audio for {duration} seconds...")
            
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=duration)
                
            # Save audio to file
            with open(output_path, "wb") as f:
                f.write(audio.get_wav_data())
                
            logger.info(f"Audio recorded to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
            
    def is_available(self) -> bool:
        """Check if voice interface is available"""
        return (
            self.recognizer is not None and 
            self.tts_engine is not None and
            self.microphone is not None
        )
        
    async def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio devices"""
        try:
            devices = []
            
            # Get PyAudio instance
            p = pyaudio.PyAudio()
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
                
            p.terminate()
            return devices
            
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return []
            
    async def set_microphone_device(self, device_index: int):
        """Set microphone device by index"""
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            
            # Recalibrate for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                
            logger.info(f"Microphone set to device index: {device_index}")
            
        except Exception as e:
            logger.error(f"Error setting microphone device: {e}")
            
    async def enhance_audio_quality(self, audio_path: str) -> str:
        """Enhance audio quality using noise reduction"""
        try:
            # Load audio
            data, samplerate = sf.read(audio_path)
            
            # Apply noise reduction (simple high-pass filter)
            sos = signal.butter(5, 300, btype='high', fs=samplerate, output='sos')
            filtered = signal.sosfilt(sos, data)
            
            # Normalize audio
            filtered = filtered / np.max(np.abs(filtered))
            
            # Save enhanced audio
            enhanced_path = audio_path.replace('.wav', '_enhanced.wav')
            sf.write(enhanced_path, filtered, samplerate)
            
            logger.info(f"Audio enhanced and saved to: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_path
            
    async def get_voice_metrics(self) -> Dict[str, Any]:
        """Get voice interface performance metrics"""
        return {
            'is_listening': self.is_listening,
            'is_recording': self.is_recording,
            'continuous_mode': self.continuous_mode,
            'wake_words_enabled': self.wake_word_detector['enabled'] if self.wake_word_detector else False,
            'recognition_language': self.recognition_language,
            'microphone_available': self.microphone is not None,
            'tts_available': self.tts_engine is not None,
            'whisper_available': self.whisper_pipeline is not None
        }