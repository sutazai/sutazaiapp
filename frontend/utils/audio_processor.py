"""
Audio Processing Utilities Module
Handles audio format conversion, noise reduction, and processing
"""

import numpy as np
import wave
import io
import audioop
import struct
from typing import Tuple, Optional, Union, Dict, List
import threading
import queue
import time
from collections import deque
import math

class AudioProcessor:
    """Advanced audio processing utilities"""
    
    def __init__(self):
        self.sample_rate = 16000  # Standard for speech recognition
        self.channels = 1  # Mono
        self.sample_width = 2  # 16-bit
        self.chunk_size = 1024
        
        # Voice Activity Detection (VAD) parameters
        self.energy_threshold = 1000
        self.silence_threshold = 500
        self.speech_threshold = 1500
        
        # Noise reduction parameters
        self.noise_profile = None
        self.noise_gate_threshold = 0.01
        
        # Audio buffer for streaming
        self.audio_buffer = deque(maxlen=100)
        self.recording_buffer = []
        
    def bytes_to_numpy(self, audio_bytes: bytes, 
                      sample_width: int = 2) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Convert bytes to numpy array
            if sample_width == 1:
                dtype = np.uint8
                offset = 128
            elif sample_width == 2:
                dtype = np.int16
                offset = 0
            elif sample_width == 4:
                dtype = np.int32
                offset = 0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            audio_array = np.frombuffer(audio_bytes, dtype=dtype)
            
            # Normalize to float32 [-1, 1]
            if sample_width == 1:
                audio_array = (audio_array.astype(np.float32) - offset) / 128.0
            else:
                max_val = 2 ** (sample_width * 8 - 1)
                audio_array = audio_array.astype(np.float32) / max_val
            
            return audio_array
        except Exception as e:
            print(f"Bytes to numpy conversion error: {e}")
            return np.array([])
    
    def numpy_to_bytes(self, audio_array: np.ndarray, 
                      sample_width: int = 2) -> bytes:
        """Convert numpy array to audio bytes"""
        try:
            # Ensure array is in [-1, 1] range
            audio_array = np.clip(audio_array, -1, 1)
            
            # Convert to appropriate integer type
            if sample_width == 1:
                audio_array = ((audio_array * 128) + 128).astype(np.uint8)
            elif sample_width == 2:
                audio_array = (audio_array * 32767).astype(np.int16)
            elif sample_width == 4:
                audio_array = (audio_array * 2147483647).astype(np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            return audio_array.tobytes()
        except Exception as e:
            print(f"Numpy to bytes conversion error: {e}")
            return b""
    
    def resample_audio(self, audio_data: np.ndarray, 
                      orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            if orig_sr == target_sr:
                return audio_data
            
            # Simple linear interpolation resampling
            duration = len(audio_data) / orig_sr
            target_length = int(duration * target_sr)
            
            indices = np.linspace(0, len(audio_data) - 1, target_length)
            resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            
            return resampled
        except Exception as e:
            print(f"Resampling error: {e}")
            return audio_data
    
    def apply_noise_gate(self, audio_data: np.ndarray, 
                        threshold: float = None) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        try:
            if threshold is None:
                threshold = self.noise_gate_threshold
            
            # Calculate envelope
            envelope = np.abs(audio_data)
            
            # Apply gate
            gated = np.where(envelope > threshold, audio_data, 0)
            
            # Smooth transitions
            return self.smooth_audio(gated)
        except Exception as e:
            print(f"Noise gate error: {e}")
            return audio_data
    
    def reduce_noise(self, audio_data: np.ndarray, 
                    noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Reduce noise using spectral subtraction"""
        try:
            if noise_profile is None:
                noise_profile = self.noise_profile
            
            if noise_profile is None:
                # Simple high-pass filter for noise reduction
                return self.apply_highpass_filter(audio_data, cutoff=100)
            
            # Spectral subtraction
            # Convert to frequency domain
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Subtract noise profile
            noise_fft = np.fft.rfft(noise_profile, n=len(audio_data))
            noise_magnitude = np.abs(noise_fft)
            
            # Ensure same shape
            min_len = min(len(magnitude), len(noise_magnitude))
            magnitude[:min_len] -= noise_magnitude[:min_len] * 0.8
            magnitude = np.maximum(magnitude, 0)
            
            # Reconstruct signal
            fft = magnitude * np.exp(1j * phase)
            cleaned = np.fft.irfft(fft, n=len(audio_data))
            
            return cleaned
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio_data
    
    def apply_highpass_filter(self, audio_data: np.ndarray, 
                             cutoff: float = 100) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        try:
            # Simple high-pass filter using FFT
            fft = np.fft.rfft(audio_data)
            frequencies = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Zero out frequencies below cutoff
            fft[frequencies < cutoff] = 0
            
            # Reconstruct signal
            filtered = np.fft.irfft(fft, n=len(audio_data))
            
            return filtered
        except Exception as e:
            print(f"High-pass filter error: {e}")
            return audio_data
    
    def normalize_volume(self, audio_data: np.ndarray, 
                        target_level: float = 0.5) -> np.ndarray:
        """Normalize audio volume"""
        try:
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms > 0:
                # Calculate scaling factor
                scale = target_level / rms
                
                # Apply scaling with clipping
                normalized = audio_data * scale
                normalized = np.clip(normalized, -1, 1)
                
                return normalized
            return audio_data
        except Exception as e:
            print(f"Volume normalization error: {e}")
            return audio_data
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Detect if audio contains voice activity"""
        try:
            # Calculate energy
            energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
            
            # Voice typically has energy above threshold and moderate zero crossing rate
            has_voice = (
                energy > self.energy_threshold / 32768 and
                0.01 < zero_crossings < 0.1
            )
            
            return has_voice
        except Exception as e:
            print(f"VAD error: {e}")
            return False
    
    def detect_silence(self, audio_data: np.ndarray) -> bool:
        """Detect if audio is silence"""
        try:
            energy = np.sqrt(np.mean(audio_data ** 2))
            return energy < self.silence_threshold / 32768
        except Exception as e:
            print(f"Silence detection error: {e}")
            return False
    
    def trim_silence(self, audio_data: np.ndarray, 
                    threshold: float = None) -> np.ndarray:
        """Trim silence from beginning and end of audio"""
        try:
            if threshold is None:
                threshold = self.silence_threshold / 32768
            
            # Find non-silent regions
            envelope = np.abs(audio_data)
            non_silent = envelope > threshold
            
            if not np.any(non_silent):
                return audio_data
            
            # Find first and last non-silent samples
            indices = np.where(non_silent)[0]
            start = max(0, indices[0] - self.chunk_size)
            end = min(len(audio_data), indices[-1] + self.chunk_size)
            
            return audio_data[start:end]
        except Exception as e:
            print(f"Trim silence error: {e}")
            return audio_data
    
    def smooth_audio(self, audio_data: np.ndarray, 
                    window_size: int = 5) -> np.ndarray:
        """Smooth audio signal to reduce artifacts"""
        try:
            # Apply moving average filter
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(audio_data, kernel, mode='same')
            return smoothed
        except Exception as e:
            print(f"Smoothing error: {e}")
            return audio_data
    
    def calculate_features(self, audio_data: np.ndarray) -> Dict:
        """Calculate audio features for analysis"""
        try:
            features = {}
            
            # Time domain features
            features["rms"] = np.sqrt(np.mean(audio_data ** 2))
            features["energy"] = np.sum(audio_data ** 2)
            features["zero_crossing_rate"] = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
            features["max_amplitude"] = np.max(np.abs(audio_data))
            
            # Frequency domain features
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            frequencies = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Spectral centroid
            if np.sum(magnitude) > 0:
                features["spectral_centroid"] = np.sum(frequencies * magnitude) / np.sum(magnitude)
            else:
                features["spectral_centroid"] = 0
            
            # Dominant frequency
            dominant_idx = np.argmax(magnitude)
            features["dominant_frequency"] = frequencies[dominant_idx]
            
            return features
        except Exception as e:
            print(f"Feature calculation error: {e}")
            return {}
    
    def create_spectrogram(self, audio_data: np.ndarray, 
                          window_size: int = 512,
                          hop_length: int = 256) -> np.ndarray:
        """Create spectrogram from audio data"""
        try:
            # Calculate number of frames
            num_frames = (len(audio_data) - window_size) // hop_length + 1
            
            # Initialize spectrogram
            spectrogram = np.zeros((window_size // 2 + 1, num_frames))
            
            # Apply window function
            window = np.hanning(window_size)
            
            # Calculate FFT for each frame
            for i in range(num_frames):
                start = i * hop_length
                end = start + window_size
                
                if end <= len(audio_data):
                    frame = audio_data[start:end] * window
                    fft = np.fft.rfft(frame)
                    spectrogram[:, i] = np.abs(fft)
            
            # Convert to dB scale
            spectrogram = 20 * np.log10(spectrogram + 1e-10)
            
            return spectrogram
        except Exception as e:
            print(f"Spectrogram error: {e}")
            return np.array([])
    
    def enhance_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance speech quality"""
        try:
            # Apply series of enhancements
            enhanced = audio_data.copy()
            
            # 1. Noise reduction
            enhanced = self.reduce_noise(enhanced)
            
            # 2. High-pass filter
            enhanced = self.apply_highpass_filter(enhanced, cutoff=80)
            
            # 3. Normalize volume
            enhanced = self.normalize_volume(enhanced, target_level=0.7)
            
            # 4. Trim silence
            enhanced = self.trim_silence(enhanced)
            
            return enhanced
        except Exception as e:
            print(f"Speech enhancement error: {e}")
            return audio_data
    
    def chunk_audio(self, audio_data: np.ndarray, 
                   chunk_duration: float = 0.5) -> List[np.ndarray]:
        """Split audio into chunks"""
        try:
            chunk_samples = int(chunk_duration * self.sample_rate)
            chunks = []
            
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                if len(chunk) == chunk_samples:
                    chunks.append(chunk)
                elif len(chunk) > chunk_samples // 2:
                    # Pad last chunk if it's more than half the size
                    padded = np.zeros(chunk_samples)
                    padded[:len(chunk)] = chunk
                    chunks.append(padded)
            
            return chunks
        except Exception as e:
            print(f"Chunking error: {e}")
            return []
    
    def merge_audio(self, chunks: List[np.ndarray], 
                   overlap: int = 0) -> np.ndarray:
        """Merge audio chunks with optional overlap"""
        try:
            if not chunks:
                return np.array([])
            
            if overlap == 0:
                return np.concatenate(chunks)
            
            # Merge with crossfade
            total_length = sum(len(c) for c in chunks) - overlap * (len(chunks) - 1)
            merged = np.zeros(total_length)
            
            position = 0
            for i, chunk in enumerate(chunks):
                if i == 0:
                    merged[position:position + len(chunk)] = chunk
                else:
                    # Apply crossfade
                    fade_in = np.linspace(0, 1, overlap)
                    fade_out = np.linspace(1, 0, overlap)
                    
                    # Overlapping region
                    merged[position - overlap:position] *= fade_out
                    merged[position - overlap:position] += chunk[:overlap] * fade_in
                    
                    # Non-overlapping region
                    merged[position:position + len(chunk) - overlap] = chunk[overlap:]
                
                position += len(chunk) - overlap
            
            return merged
        except Exception as e:
            print(f"Merge error: {e}")
            return np.array([])
    
    def convert_format(self, audio_bytes: bytes, 
                      input_format: Dict, 
                      output_format: Dict) -> bytes:
        """Convert audio between formats"""
        try:
            # Parse input format
            input_sr = input_format.get('sample_rate', 16000)
            input_channels = input_format.get('channels', 1)
            input_width = input_format.get('sample_width', 2)
            
            # Parse output format
            output_sr = output_format.get('sample_rate', 16000)
            output_channels = output_format.get('channels', 1)
            output_width = output_format.get('sample_width', 2)
            
            # Convert to numpy
            audio_array = self.bytes_to_numpy(audio_bytes, input_width)
            
            # Resample if needed
            if input_sr != output_sr:
                audio_array = self.resample_audio(audio_array, input_sr, output_sr)
            
            # Convert channels if needed
            # (Simplified - just duplicate mono to stereo or average stereo to mono)
            if input_channels != output_channels:
                if output_channels == 2 and input_channels == 1:
                    # Mono to stereo
                    audio_array = np.repeat(audio_array, 2)
                elif output_channels == 1 and input_channels == 2:
                    # Stereo to mono
                    audio_array = audio_array[::2]  # Take every other sample
            
            # Convert back to bytes
            return self.numpy_to_bytes(audio_array, output_width)
        except Exception as e:
            print(f"Format conversion error: {e}")
            return audio_bytes
    
    def save_audio(self, audio_data: Union[bytes, np.ndarray], 
                  filename: str, sample_rate: Optional[int] = None,
                  channels: int = 1, sample_width: int = 2):
        """Save audio data to WAV file"""
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # Convert numpy to bytes if needed
            if isinstance(audio_data, np.ndarray):
                audio_data = self.numpy_to_bytes(audio_data, sample_width)
            
            # Write WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            return True
        except Exception as e:
            print(f"Save audio error: {e}")
            return False
    
    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        """Load audio from WAV file"""
        try:
            with wave.open(filename, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
            
            audio_array = self.bytes_to_numpy(frames, sample_width)
            return audio_array, sample_rate
        except Exception as e:
            print(f"Load audio error: {e}")
            return np.array([]), 0


class AudioStreamProcessor:
    """Real-time audio stream processing"""
    
    def __init__(self, callback=None):
        self.processor = AudioProcessor()
        self.callback = callback
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        
    def start_stream(self):
        """Start audio stream processing"""
        if not self.is_recording:
            self.is_recording = True
            self.processing_thread = threading.Thread(
                target=self._process_stream,
                daemon=True
            )
            self.processing_thread.start()
    
    def stop_stream(self):
        """Stop audio stream processing"""
        self.is_recording = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
    
    def add_audio_chunk(self, chunk: bytes):
        """Add audio chunk to processing queue"""
        if self.is_recording:
            self.audio_queue.put(chunk)
    
    def _process_stream(self):
        """Process audio stream in real-time"""
        buffer = []
        
        while self.is_recording:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Convert to numpy
                audio_array = self.processor.bytes_to_numpy(chunk)
                
                # Process chunk
                processed = self.processor.enhance_speech(audio_array)
                
                # Check for voice activity
                if self.processor.detect_voice_activity(processed):
                    buffer.append(processed)
                elif buffer:
                    # End of speech detected
                    complete_audio = np.concatenate(buffer)
                    
                    if self.callback:
                        self.callback(complete_audio)
                    
                    buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Stream processing error: {e}")