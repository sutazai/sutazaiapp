"""
Voice Assistant Module
Handles speech recognition and text-to-speech functionality
"""

import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import numpy as np
from typing import Optional, Dict, Any
import io
import wave
import audioop

class VoiceAssistant:
    """Advanced voice assistant with wake word detection and natural speech"""
    
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self._configure_voice()
        
        # Threading for background listening
        self.listening_thread = None
        self.is_listening = False
        self.command_queue = queue.Queue()
        
        # Wake word configuration
        self.wake_words = ["jarvis", "hey jarvis", "ok jarvis"]
        self.wake_word_detected = False
        
        # Audio processing
        self.sample_rate = 16000
        self.chunk_size = 1024
        
    def _configure_voice(self):
        """Configure TTS voice parameters"""
        voices = self.engine.getProperty('voices')
        
        # Try to set a male voice (JARVIS-like)
        for voice in voices:
            if "male" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        # Set speech parameters
        self.engine.setProperty('rate', 175)  # Speaking rate
        self.engine.setProperty('volume', 1.0)  # Volume
        
    def start_listening(self, callback=None):
        """Start continuous background listening"""
        if not self.is_listening:
            self.is_listening = True
            self.listening_thread = threading.Thread(
                target=self._listen_background,
                args=(callback,),
                daemon=True
            )
            self.listening_thread.start()
            return True
        return False
    
    def stop_listening(self):
        """Stop background listening"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=1)
            self.listening_thread = None
    
    def _listen_background(self, callback):
        """Background listening thread"""
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
        while self.is_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    # Short timeout for responsiveness
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1,
                        phrase_time_limit=5
                    )
                
                # Process audio
                text = self.audio_to_text(audio)
                if text:
                    # Check for wake word
                    if self._detect_wake_word(text):
                        self.wake_word_detected = True
                        if callback:
                            callback("wake_word_detected", text)
                    elif self.wake_word_detected:
                        # Process command after wake word
                        self.command_queue.put(text)
                        if callback:
                            callback("command", text)
                        self.wake_word_detected = False
                        
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Listening error: {e}")
    
    def _detect_wake_word(self, text: str) -> bool:
        """Check if text contains wake word"""
        text_lower = text.lower()
        return any(wake_word in text_lower for wake_word in self.wake_words)
    
    def audio_to_text(self, audio_data) -> Optional[str]:
        """Convert audio to text using multiple recognition engines"""
        try:
            # Try Google Speech Recognition first
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            # Try alternative: Sphinx (offline)
            try:
                text = self.recognizer.recognize_sphinx(audio_data)
                return text
            except:
                return None
        except sr.RequestError as e:
            print(f"Recognition service error: {e}")
            return None
        except Exception as e:
            print(f"Audio to text error: {e}")
            return None
    
    def speak(self, text: str, wait: bool = True):
        """Convert text to speech"""
        try:
            if wait:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Non-blocking speech
                threading.Thread(
                    target=self._speak_async,
                    args=(text,),
                    daemon=True
                ).start()
        except Exception as e:
            print(f"TTS error: {e}")
    
    def _speak_async(self, text: str):
        """Asynchronous speech synthesis"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def process_audio_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Process raw audio bytes and convert to text"""
        try:
            # Convert bytes to AudioData
            audio_io = io.BytesIO(audio_bytes)
            with wave.open(audio_io, 'rb') as wav_file:
                # Get audio parameters
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                
            # Create AudioData object
            audio_data = sr.AudioData(frames, sample_rate, sample_width)
            
            # Convert to text
            return self.audio_to_text(audio_data)
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    def get_audio_level(self, audio_data) -> float:
        """Get current audio level for visualization"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) for volume level
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Normalize to 0-100 range
            max_val = 32768  # Max value for 16-bit audio
            level = (rms / max_val) * 100
            
            return min(100, level)
        except Exception as e:
            print(f"Audio level error: {e}")
            return 0.0
    
    def calibrate_microphone(self) -> Dict[str, float]:
        """Calibrate microphone for optimal recognition"""
        calibration = {
            "ambient_noise": 0.0,
            "energy_threshold": 0.0,
            "recommended_gain": 1.0
        }
        
        try:
            with self.microphone as source:
                print("Calibrating microphone... Please remain silent.")
                
                # Measure ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
                calibration["ambient_noise"] = self.recognizer.energy_threshold
                
                # Set energy threshold slightly above ambient
                calibration["energy_threshold"] = self.recognizer.energy_threshold * 1.2
                self.recognizer.energy_threshold = calibration["energy_threshold"]
                
                # Calculate recommended gain
                if calibration["ambient_noise"] < 100:
                    calibration["recommended_gain"] = 1.5
                elif calibration["ambient_noise"] < 300:
                    calibration["recommended_gain"] = 1.2
                else:
                    calibration["recommended_gain"] = 1.0
                
                print("Calibration complete!")
                
        except Exception as e:
            print(f"Calibration error: {e}")
        
        return calibration
    
    def set_voice_parameters(self, rate: int = 175, volume: float = 1.0, voice_index: int = 0):
        """Update voice parameters"""
        try:
            voices = self.engine.getProperty('voices')
            if voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
            
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
        except Exception as e:
            print(f"Voice parameter error: {e}")
    
    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        try:
            voices = self.engine.getProperty('voices')
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": voice.languages,
                    "gender": "male" if "male" in voice.name.lower() else "female"
                }
                for voice in voices
            ]
        except Exception as e:
            print(f"Voice list error: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        if self.engine:
            self.engine.stop()