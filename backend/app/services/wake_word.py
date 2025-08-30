"""
Wake Word Detection Service for JARVIS
Implements multiple wake word detection methods with fallback
"""

import asyncio
import json
import logging
import struct
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Try to import various wake word detection libraries
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    logging.warning("Porcupine not available for wake word detection")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logging.warning("SpeechRecognition not available")

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available")

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for neural wake word detection")

logger = logging.getLogger(__name__)


class WakeWordEngine(Enum):
    """Available wake word detection engines"""
    PORCUPINE = "porcupine"  # High accuracy, requires license
    VOSK = "vosk"  # Free, offline, decent accuracy
    SPEECH_RECOGNITION = "speech_recognition"  # Simple but less efficient
    NEURAL = "neural"  # Custom neural network (if model available)
    ENERGY_BASED = "energy_based"  # Simple energy-based detection


@dataclass
class WakeWordConfig:
    """Wake word detection configuration"""
    engine: WakeWordEngine = WakeWordEngine.ENERGY_BASED
    keywords: List[str] = None
    sensitivity: float = 0.5  # 0.0 to 1.0
    threshold: float = 0.7  # Confidence threshold
    buffer_size: int = 50  # Audio buffer size for detection
    sample_rate: int = 16000
    frame_length: int = 512
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = ["jarvis", "hey jarvis", "ok jarvis", "hello jarvis"]


@dataclass
class WakeWordDetection:
    """Wake word detection result"""
    detected: bool
    keyword: Optional[str]
    confidence: float
    timestamp: float
    audio_segment: Optional[bytes]


class WakeWordDetector:
    """
    Multi-engine wake word detector with fallback support
    Implements various detection methods from different JARVIS repositories
    """
    
    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()
        self.is_active = False
        
        # Initialize detection engines
        self.porcupine = None
        self.vosk_model = None
        self.recognizer = None
        self.neural_model = None
        
        # Audio buffer for detection
        self.audio_buffer = deque(maxlen=self.config.buffer_size)
        
        # Detection state
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        
        # Metrics
        self.metrics = {
            "total_detections": 0,
            "false_positives": 0,
            "true_positives": 0,
            "engine_usage": {},
            "average_confidence": 0
        }
        
        # Initialize selected engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected wake word detection engine"""
        engine = self.config.engine
        
        if engine == WakeWordEngine.PORCUPINE and PORCUPINE_AVAILABLE:
            self._initialize_porcupine()
        elif engine == WakeWordEngine.VOSK and VOSK_AVAILABLE:
            self._initialize_vosk()
        elif engine == WakeWordEngine.SPEECH_RECOGNITION and SR_AVAILABLE:
            self._initialize_speech_recognition()
        elif engine == WakeWordEngine.NEURAL and TORCH_AVAILABLE:
            self._initialize_neural()
        else:
            # Default to energy-based detection
            logger.info("Using energy-based wake word detection")
            self.config.engine = WakeWordEngine.ENERGY_BASED
    
    def _initialize_porcupine(self):
        """Initialize Porcupine wake word engine"""
        try:
            # Note: Requires Porcupine access key
            # This is a placeholder - actual implementation needs valid key
            access_key = "YOUR_PORCUPINE_ACCESS_KEY"
            
            keyword_paths = []
            sensitivities = []
            
            # Try to use built-in keywords
            for keyword in self.config.keywords:
                if keyword.lower() in ["jarvis", "computer", "alexa", "ok google"]:
                    keyword_paths.append(keyword.lower())
                    sensitivities.append(self.config.sensitivity)
            
            if keyword_paths:
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=keyword_paths,
                    sensitivities=sensitivities
                )
                logger.info(f"Porcupine initialized with keywords: {keyword_paths}")
            else:
                logger.warning("No valid keywords for Porcupine")
                
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            self.porcupine = None
    
    def _initialize_vosk(self):
        """Initialize Vosk for wake word detection"""
        try:
            # Try to load small English model
            model_path = Path("/opt/models/vosk-model-small-en-us-0.15")
            
            if not model_path.exists():
                # Try alternative paths
                alternative_paths = [
                    Path.home() / ".cache" / "vosk" / "vosk-model-small-en-us-0.15",
                    Path("/usr/share/vosk/models/vosk-model-small-en-us-0.15"),
                    Path("./models/vosk-model-small-en-us-0.15")
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        break
            
            if model_path.exists():
                self.vosk_model = vosk.Model(str(model_path))
                self.vosk_recognizer = vosk.KaldiRecognizer(
                    self.vosk_model,
                    self.config.sample_rate
                )
                
                # Set grammar for wake words
                grammar = {
                    "phrases": self.config.keywords
                }
                self.vosk_recognizer.SetGrammar(json.dumps(grammar))
                
                logger.info("Vosk wake word detector initialized")
            else:
                logger.warning(f"Vosk model not found at {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            self.vosk_model = None
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition for wake word detection"""
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.pause_threshold = 0.5
            logger.info("Speech recognition wake word detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
            self.recognizer = None
    
    def _initialize_neural(self):
        """Initialize neural network for wake word detection"""
        try:
            # Load a pre-trained model if available
            model_path = Path("/opt/models/wake_word_model.pt")
            
            if model_path.exists():
                self.neural_model = torch.load(model_path)
                self.neural_model.eval()
                logger.info("Neural wake word model loaded")
            else:
                logger.warning("Neural wake word model not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize neural model: {e}")
            self.neural_model = None
    
    async def detect(self, audio_data: bytes) -> WakeWordDetection:
        """
        Detect wake word in audio data
        Returns detection result with confidence
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            return WakeWordDetection(
                detected=False,
                keyword=None,
                confidence=0.0,
                timestamp=current_time,
                audio_segment=None
            )
        
        # Add to buffer
        self.audio_buffer.append(audio_data)
        
        # Perform detection based on engine
        result = None
        
        if self.config.engine == WakeWordEngine.PORCUPINE and self.porcupine:
            result = await self._detect_porcupine(audio_data)
        elif self.config.engine == WakeWordEngine.VOSK and self.vosk_model:
            result = await self._detect_vosk(audio_data)
        elif self.config.engine == WakeWordEngine.SPEECH_RECOGNITION and self.recognizer:
            result = await self._detect_speech_recognition(audio_data)
        elif self.config.engine == WakeWordEngine.NEURAL and self.neural_model:
            result = await self._detect_neural(audio_data)
        else:
            result = await self._detect_energy_based(audio_data)
        
        # Update metrics if detected
        if result and result.detected:
            self.last_detection_time = current_time
            self.metrics["total_detections"] += 1
            self._update_confidence_metric(result.confidence)
            
            # Track engine usage
            engine_name = self.config.engine.value
            self.metrics["engine_usage"][engine_name] = \
                self.metrics["engine_usage"].get(engine_name, 0) + 1
        
        return result or WakeWordDetection(
            detected=False,
            keyword=None,
            confidence=0.0,
            timestamp=current_time,
            audio_segment=None
        )
    
    async def _detect_porcupine(self, audio_data: bytes) -> Optional[WakeWordDetection]:
        """Detect wake word using Porcupine"""
        try:
            # Convert audio to format Porcupine expects
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) >= self.porcupine.frame_length:
                keyword_index = self.porcupine.process(
                    audio_array[:self.porcupine.frame_length]
                )
                
                if keyword_index >= 0:
                    keyword = self.config.keywords[keyword_index]
                    return WakeWordDetection(
                        detected=True,
                        keyword=keyword,
                        confidence=0.95,  # Porcupine is very confident
                        timestamp=time.time(),
                        audio_segment=audio_data
                    )
                    
        except Exception as e:
            logger.error(f"Porcupine detection error: {e}")
        
        return None
    
    async def _detect_vosk(self, audio_data: bytes) -> Optional[WakeWordDetection]:
        """Detect wake word using Vosk"""
        try:
            if self.vosk_recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get("text", "").lower()
                
                # Check for wake words
                for keyword in self.config.keywords:
                    if keyword.lower() in text:
                        confidence = result.get("confidence", 0.5)
                        
                        if confidence >= self.config.threshold:
                            return WakeWordDetection(
                                detected=True,
                                keyword=keyword,
                                confidence=confidence,
                                timestamp=time.time(),
                                audio_segment=audio_data
                            )
                            
        except Exception as e:
            logger.error(f"Vosk detection error: {e}")
        
        return None
    
    async def _detect_speech_recognition(self, audio_data: bytes) -> Optional[WakeWordDetection]:
        """Detect wake word using speech recognition"""
        try:
            # Accumulate some audio before trying recognition
            if len(self.audio_buffer) < 10:
                return None
            
            # Combine recent audio
            combined_audio = b''.join(list(self.audio_buffer)[-10:])
            
            # Convert to AudioData
            audio = sr.AudioData(combined_audio, self.config.sample_rate, 2)
            
            # Try recognition with timeout
            try:
                text = self.recognizer.recognize_google(
                    audio,
                    language="en-US",
                    show_all=False
                ).lower()
                
                # Check for wake words
                for keyword in self.config.keywords:
                    if keyword.lower() in text:
                        return WakeWordDetection(
                            detected=True,
                            keyword=keyword,
                            confidence=0.7,  # Medium confidence
                            timestamp=time.time(),
                            audio_segment=combined_audio
                        )
                        
            except (sr.UnknownValueError, sr.RequestError):
                pass
                
        except Exception as e:
            logger.error(f"Speech recognition detection error: {e}")
        
        return None
    
    async def _detect_neural(self, audio_data: bytes) -> Optional[WakeWordDetection]:
        """Detect wake word using neural network"""
        try:
            # Convert audio to tensor
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = self.neural_model(audio_tensor)
                confidence = torch.sigmoid(output).item()
            
            if confidence >= self.config.threshold:
                return WakeWordDetection(
                    detected=True,
                    keyword=self.config.keywords[0],  # Primary keyword
                    confidence=confidence,
                    timestamp=time.time(),
                    audio_segment=audio_data
                )
                
        except Exception as e:
            logger.error(f"Neural detection error: {e}")
        
        return None
    
    async def _detect_energy_based(self, audio_data: bytes) -> Optional[WakeWordDetection]:
        """
        Simple energy-based detection
        Detects when audio energy exceeds threshold (voice activity)
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate energy
            energy = np.sqrt(np.mean(audio_array ** 2))
            
            # Simple threshold (adjust based on environment)
            energy_threshold = 1000
            
            if energy > energy_threshold:
                # Voice activity detected
                # In real implementation, would need more sophisticated detection
                return WakeWordDetection(
                    detected=True,
                    keyword="[voice_activity]",
                    confidence=min(energy / 5000, 1.0),
                    timestamp=time.time(),
                    audio_segment=audio_data
                )
                
        except Exception as e:
            logger.error(f"Energy-based detection error: {e}")
        
        return None
    
    def _update_confidence_metric(self, confidence: float):
        """Update average confidence metric"""
        total = self.metrics["total_detections"]
        if total > 0:
            avg = self.metrics["average_confidence"]
            self.metrics["average_confidence"] = (avg * (total - 1) + confidence) / total
    
    def mark_true_positive(self):
        """Mark last detection as true positive"""
        self.metrics["true_positives"] += 1
    
    def mark_false_positive(self):
        """Mark last detection as false positive"""
        self.metrics["false_positives"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detection metrics"""
        metrics = self.metrics.copy()
        
        # Calculate precision if we have data
        total_positives = metrics["true_positives"] + metrics["false_positives"]
        if total_positives > 0:
            metrics["precision"] = metrics["true_positives"] / total_positives
        else:
            metrics["precision"] = 0.0
        
        return metrics
    
    def set_keywords(self, keywords: List[str]):
        """Update wake word keywords"""
        self.config.keywords = keywords
        
        # Reinitialize engines with new keywords
        if self.config.engine == WakeWordEngine.VOSK and self.vosk_recognizer:
            grammar = {"phrases": keywords}
            self.vosk_recognizer.SetGrammar(json.dumps(grammar))
    
    def set_sensitivity(self, sensitivity: float):
        """Update detection sensitivity (0.0 to 1.0)"""
        self.config.sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Update engine-specific sensitivity
        if self.porcupine:
            # Porcupine doesn't support runtime sensitivity changes
            # Would need to reinitialize
            pass
    
    async def calibrate(self, background_audio: bytes, duration: float = 5.0) -> bool:
        """
        Calibrate detector with background noise
        Adjusts thresholds based on ambient noise level
        """
        try:
            # Calculate background noise level
            audio_array = np.frombuffer(background_audio, dtype=np.int16)
            noise_level = np.sqrt(np.mean(audio_array ** 2))
            
            # Adjust threshold based on noise
            # Set threshold to be 2-3x the noise level
            self.config.threshold = min(0.9, max(0.3, noise_level * 2.5 / 5000))
            
            logger.info(f"Calibrated with noise level: {noise_level}, threshold: {self.config.threshold}")
            return True
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        self.vosk_model = None
        self.neural_model = None
        self.recognizer = None
        
        logger.info("Wake word detector cleaned up")


# Singleton instance
_detector_instance = None


def get_wake_word_detector(config: Optional[WakeWordConfig] = None) -> WakeWordDetector:
    """Get or create wake word detector singleton"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = WakeWordDetector(config)
    return _detector_instance