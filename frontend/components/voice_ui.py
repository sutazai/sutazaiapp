"""
Voice UI Components for JARVIS Frontend
Handles voice recording, visualization, and wake word detection
"""

import streamlit as st
import numpy as np
import sounddevice as sd
import wave
import io
import threading
import queue
import time
from typing import Optional, Callable, Tuple
import webrtc_streamer as webrtc
from dataclasses import dataclass
import plotly.graph_objects as go

@dataclass
class VoiceConfig:
    """Voice configuration settings"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    wake_word: str = "hey jarvis"
    vad_aggressiveness: int = 3

class VoiceVisualizer:
    """Real-time audio visualization component"""
    
    def __init__(self, width: int = 400, height: int = 100):
        self.width = width
        self.height = height
        self.history_size = 50
        self.amplitude_history = [0] * self.history_size
    
    def update(self, audio_data: np.ndarray) -> go.Figure:
        """Update visualization with new audio data"""
        # Calculate amplitude
        amplitude = np.abs(audio_data).mean()
        
        # Update history
        self.amplitude_history.pop(0)
        self.amplitude_history.append(amplitude)
        
        # Create visualization
        fig = go.Figure()
        
        # Add waveform
        fig.add_trace(go.Scatter(
            y=self.amplitude_history,
            mode='lines',
            line=dict(color='#00D4FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.2)'
        ))
        
        # Update layout
        fig.update_layout(
            width=self.width,
            height=self.height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, range=[0, 0.5])
        )
        
        return fig

class VoiceRecorder:
    """Advanced voice recording with VAD and wake word detection"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.recording = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.silence_start = None
        self.stream = None
        self.thread = None
    
    def start(self, callback: Optional[Callable] = None):
        """Start recording audio"""
        self.recording = True
        self.audio_buffer = []
        self.silence_start = None
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            
            if self.recording:
                # Add to queue for processing
                self.audio_queue.put(indata.copy())
                
                # Check for silence (simple VAD)
                amplitude = np.abs(indata).mean()
                
                if amplitude < self.config.silence_threshold:
                    if self.silence_start is None:
                        self.silence_start = time.time()
                    elif time.time() - self.silence_start > self.config.silence_duration:
                        # Auto-stop on extended silence
                        if callback:
                            callback('silence_detected')
                else:
                    self.silence_start = None
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            callback=audio_callback,
            blocksize=self.config.chunk_size
        )
        self.stream.start()
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.start()
    
    def _process_audio(self):
        """Process audio from queue"""
        while self.recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.append(audio_chunk)
            except queue.Empty:
                continue
    
    def stop(self) -> bytes:
        """Stop recording and return audio data"""
        self.recording = False
        
        # Stop stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # Wait for processing thread
        if self.thread:
            self.thread.join(timeout=1)
        
        # Convert buffer to WAV
        if not self.audio_buffer:
            return b''
        
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        
        # Create WAV file in memory
        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            return buffer.getvalue()
    
    def get_current_amplitude(self) -> float:
        """Get current audio amplitude for visualization"""
        if not self.audio_buffer:
            return 0.0
        
        recent_audio = self.audio_buffer[-1] if self.audio_buffer else np.zeros(1)
        return np.abs(recent_audio).mean()

class WakeWordDetector:
    """Simple wake word detection (placeholder for more advanced implementation)"""
    
    def __init__(self, wake_word: str = "hey jarvis"):
        self.wake_word = wake_word.lower()
        self.active = False
        self.last_detection = None
    
    def process_text(self, text: str) -> bool:
        """Check if text contains wake word"""
        if not text:
            return False
        
        text_lower = text.lower()
        if self.wake_word in text_lower:
            self.active = True
            self.last_detection = time.time()
            return True
        
        # Deactivate after 30 seconds
        if self.last_detection and time.time() - self.last_detection > 30:
            self.active = False
        
        return False
    
    def is_active(self) -> bool:
        """Check if wake word is currently active"""
        if self.last_detection and time.time() - self.last_detection > 30:
            self.active = False
        return self.active

def create_voice_interface() -> Tuple[bool, Optional[bytes]]:
    """Create the voice interface UI component"""
    
    # Initialize components
    if 'voice_recorder' not in st.session_state:
        config = VoiceConfig()
        st.session_state.voice_recorder = VoiceRecorder(config)
        st.session_state.voice_visualizer = VoiceVisualizer()
        st.session_state.wake_detector = WakeWordDetector()
    
    recorder = st.session_state.voice_recorder
    visualizer = st.session_state.voice_visualizer
    wake_detector = st.session_state.wake_detector
    
    # Voice control container
    voice_container = st.container()
    
    with voice_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Wake word indicator
            if wake_detector.is_active():
                st.markdown("""
                <div style="padding: 5px; background: #4CAF50; border-radius: 5px; text-align: center;">
                    <span style="color: white;">Wake Word Active</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="padding: 5px; background: #666; border-radius: 5px; text-align: center;">
                    <span style="color: #ccc;">Say "Hey Jarvis"</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Recording button
            if st.button("🎤 Start Recording" if not recorder.recording else "⏹️ Stop Recording",
                        key="voice_record_btn",
                        use_container_width=True):
                
                if not recorder.recording:
                    # Start recording
                    recorder.start(callback=lambda x: st.info(f"Event: {x}"))
                    return True, None
                else:
                    # Stop recording
                    audio_data = recorder.stop()
                    return False, audio_data
        
        with col3:
            # Audio level indicator
            if recorder.recording:
                amplitude = recorder.get_current_amplitude()
                level = min(int(amplitude * 100), 100)
                
                st.markdown(f"""
                <div style="padding: 5px; text-align: center;">
                    <div style="width: 100%; height: 20px; background: #333; border-radius: 10px;">
                        <div style="width: {level}%; height: 100%; background: {'#4CAF50' if level < 70 else '#FF6B6B'}; border-radius: 10px;"></div>
                    </div>
                    <span style="color: #999; font-size: 0.8rem;">Audio Level</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Visualization
    if recorder.recording:
        placeholder = st.empty()
        
        # Update visualization
        while recorder.recording:
            amplitude = recorder.get_current_amplitude()
            audio_sample = np.random.randn(100) * amplitude  # Simulated for visualization
            
            fig = visualizer.update(audio_sample)
            placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            time.sleep(0.1)
    
    return recorder.recording, None

def create_voice_button(key: str = "voice_btn") -> Optional[bytes]:
    """Create a simple voice recording button"""
    
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    
    if 'recorder' not in st.session_state:
        config = VoiceConfig()
        st.session_state.recorder = VoiceRecorder(config)
    
    recorder = st.session_state.recorder
    
    # Custom CSS for the button
    st.markdown("""
    <style>
    .voice-btn {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .voice-btn:hover {
        transform: scale(1.1);
    }
    
    .voice-btn.recording {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Voice button
    button_label = "🎤" if not st.session_state.recording else "⏹️"
    button_class = "voice-btn" + (" recording" if st.session_state.recording else "")
    
    if st.button(button_label, key=key, help="Click to record voice"):
        st.session_state.recording = not st.session_state.recording
        
        if st.session_state.recording:
            # Start recording
            recorder.start()
            return None
        else:
            # Stop recording and return audio
            audio_data = recorder.stop()
            return audio_data
    
    return None