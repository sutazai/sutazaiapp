#!/usr/bin/env python3
"""
SutazAI Enhanced Chat Interface with RealtimeSTT
Real-time intelligent chatbot with superior voice recognition using RealtimeSTT
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import threading
import queue
import sys
import os

# Configure Streamlit page
st.set_page_config(
    page_title="SutazAI RealtimeSTT Chat",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced chat interface
st.markdown("""
<style>
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .system-message {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    .voice-button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        margin: 10px;
        transition: all 0.3s ease;
    }
    .voice-button:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
    }
    .voice-button.listening {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        animation: pulse 2s infinite;
    }
    .voice-button.recording {
        background: linear-gradient(45deg, #ff4444, #cc0000);
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }
        50% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
    }
    .realtime-status {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 15px 0;
        font-weight: bold;
    }
    .realtime-controls {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 20px 0;
    }
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .enter-hint {
        color: #666;
        font-size: 12px;
        margin-top: 5px;
        font-style: italic;
    }
    .voice-status {
        text-align: center;
        padding: 10px;
        background: #e8f5e8;
        border-radius: 10px;
        margin: 10px 0;
    }
    .realtime-transcript {
        background: #f0f8ff;
        border: 2px solid #4169e1;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
        min-height: 60px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Add JavaScript for Enter key functionality and microphone permissions
st.markdown("""
<script>
// Microphone permission handling
let microphonePermission = false;
let microphoneStream = null;

async function requestMicrophonePermission() {
    try {
        console.log('Requesting microphone permission...');
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        microphoneStream = stream;
        microphonePermission = true;
        console.log('Microphone permission granted!');
        
        // Show permission granted message
        const permissionStatus = document.getElementById('mic-permission-status');
        if (permissionStatus) {
            permissionStatus.innerHTML = '✅ Microphone Permission: GRANTED';
            permissionStatus.style.color = 'green';
        }
        
        // Enable voice controls
        const voiceControls = document.querySelectorAll('.voice-control-button');
        voiceControls.forEach(button => {
            button.disabled = false;
            button.style.opacity = '1';
        });
        
        return true;
    } catch (error) {
        console.error('Microphone permission denied:', error);
        microphonePermission = false;
        
        // Show permission denied message
        const permissionStatus = document.getElementById('mic-permission-status');
        if (permissionStatus) {
            permissionStatus.innerHTML = '❌ Microphone Permission: DENIED - Please allow microphone access';
            permissionStatus.style.color = 'red';
        }
        
        return false;
    }
}

// Check microphone permission status
async function checkMicrophonePermission() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            const permission = await navigator.permissions.query({ name: 'microphone' });
            console.log('Microphone permission status:', permission.state);
            
            if (permission.state === 'granted') {
                microphonePermission = true;
                const permissionStatus = document.getElementById('mic-permission-status');
                if (permissionStatus) {
                    permissionStatus.innerHTML = '✅ Microphone Permission: GRANTED';
                    permissionStatus.style.color = 'green';
                }
            } else if (permission.state === 'prompt') {
                // Will request permission when needed
                const permissionStatus = document.getElementById('mic-permission-status');
                if (permissionStatus) {
                    permissionStatus.innerHTML = '⚠️ Microphone Permission: PENDING - Click to allow';
                    permissionStatus.style.color = 'orange';
                }
            } else {
                const permissionStatus = document.getElementById('mic-permission-status');
                if (permissionStatus) {
                    permissionStatus.innerHTML = '❌ Microphone Permission: DENIED';
                    permissionStatus.style.color = 'red';
                }
            }
        } catch (error) {
            console.error('Error checking microphone permission:', error);
        }
    } else {
        console.error('MediaDevices API not supported');
        const permissionStatus = document.getElementById('mic-permission-status');
        if (permissionStatus) {
            permissionStatus.innerHTML = '❌ Microphone API not supported in this browser';
            permissionStatus.style.color = 'red';
        }
    }
}

// Auto-request microphone permission on page load
function autoRequestMicrophone() {
    setTimeout(async () => {
        if (!microphonePermission) {
            console.log('Auto-requesting microphone permission...');
            await requestMicrophonePermission();
        }
    }, 1000);
}

// Enter key functionality
function setupEnterKeyListener() {
    document.removeEventListener('keydown', handleKeyDown);
    document.addEventListener('keydown', handleKeyDown);
    
    function handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            const activeElement = document.activeElement;
            if (activeElement && activeElement.tagName === 'TEXTAREA') {
                const sendButton = document.querySelector('button[kind="primary"]');
                if (sendButton) {
                    event.preventDefault();
                    sendButton.click();
                }
            }
        }
    }
}

// Initialize everything
document.addEventListener('DOMContentLoaded', function() {
    setupEnterKeyListener();
    checkMicrophonePermission();
    autoRequestMicrophone();
});

window.addEventListener('load', function() {
    setupEnterKeyListener();
    checkMicrophonePermission();
});

setInterval(setupEnterKeyListener, 1000);
setInterval(checkMicrophonePermission, 5000);

// Global function to request microphone permission
window.requestMicrophonePermission = requestMicrophonePermission;
</script>
""", unsafe_allow_html=True)

# Backend URL
BACKEND_URLS = ["http://localhost:8000", "http://192.168.131.128:8000"]

def get_backend_url():
    """Find working backend URL"""
    for url in BACKEND_URLS:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return url
        except:
            continue
    return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = get_backend_url()
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False
if 'realtime_stt' not in st.session_state:
    st.session_state.realtime_stt = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'transcript_queue' not in st.session_state:
    st.session_state.transcript_queue = queue.Queue()
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""
if 'realtime_enabled' not in st.session_state:
    st.session_state.realtime_enabled = True
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True
if 'continuous_mode' not in st.session_state:
    st.session_state.continuous_mode = True
if 'auto_start' not in st.session_state:
    st.session_state.auto_start = True
if 'mic_permission_requested' not in st.session_state:
    st.session_state.mic_permission_requested = False

def init_realtime_stt():
    """Initialize RealtimeSTT"""
    try:
        from RealtimeSTT import AudioToTextRecorder
        
        # Configuration for RealtimeSTT with correct parameters
        recorder = AudioToTextRecorder(
            model="tiny.en",  # Fast model for real-time
            language="en",
            spinner=False,
            use_microphone=True,
            post_speech_silence_duration=0.7,
            min_length_of_recording=0.1,
            min_gap_between_recordings=0.1,
            enable_realtime_transcription=True,
            on_recording_start=lambda: st.session_state.transcript_queue.put(("recording_start", "")),
            on_recording_stop=lambda: st.session_state.transcript_queue.put(("recording_stop", "")),
            on_transcription_start=lambda: st.session_state.transcript_queue.put(("transcription_start", "")),
            on_realtime_transcription_update=lambda text: st.session_state.transcript_queue.put(("realtime_update", text)),
        )
        
        return recorder
    except ImportError:
        st.error("RealtimeSTT not properly installed. Please install it with: pip install RealtimeSTT")
        return None
    except Exception as e:
        st.error(f"Failed to initialize RealtimeSTT: {e}")
        return None

def start_realtime_recording():
    """Start RealtimeSTT recording"""
    if st.session_state.realtime_stt is None:
        st.session_state.realtime_stt = init_realtime_stt()
    
    if st.session_state.realtime_stt:
        try:
            def recording_thread():
                st.session_state.is_recording = True
                text = st.session_state.realtime_stt.text()
                st.session_state.transcript_queue.put(("final_text", text))
                st.session_state.is_recording = False
                
                # Auto-restart recording in continuous mode
                if st.session_state.continuous_mode and text.strip():
                    # Small delay before restarting
                    threading.Timer(2.0, start_realtime_recording).start()
            
            thread = threading.Thread(target=recording_thread)
            thread.daemon = True
            thread.start()
            
            return True
        except Exception as e:
            st.error(f"Failed to start recording: {e}")
            return False
    return False

def stop_realtime_recording():
    """Stop RealtimeSTT recording"""
    if st.session_state.realtime_stt:
        try:
            st.session_state.realtime_stt.stop()
            st.session_state.is_recording = False
            return True
        except Exception as e:
            st.error(f"Failed to stop recording: {e}")
            return False
    return False

def process_transcript_queue():
    """Process transcript updates from queue"""
    while not st.session_state.transcript_queue.empty():
        try:
            event_type, text = st.session_state.transcript_queue.get_nowait()
            
            if event_type == "realtime_update":
                st.session_state.current_transcript = text
            elif event_type == "stabilized":
                st.session_state.current_transcript = text
            elif event_type == "final_text":
                if text.strip():
                    st.session_state.last_message = text.strip()
                    st.session_state.current_transcript = ""
                    # Auto-process the message in continuous mode
                    if st.session_state.continuous_mode:
                        process_voice_message(text.strip())
                    st.rerun()
            elif event_type == "recording_start":
                st.session_state.current_transcript = "🎤 Recording started..."
            elif event_type == "recording_stop":
                st.session_state.current_transcript = "🧠 Processing speech..."
            elif event_type == "transcription_start":
                st.session_state.current_transcript = "📝 Transcribing..."
                
        except queue.Empty:
            break
        except Exception as e:
            st.error(f"Error processing transcript: {e}")

def make_chat_request(message: str, model: str = "llama3.2:1b"):
    """Make chat request to backend"""
    if not st.session_state.backend_url:
        return {"error": "Backend service not available"}
    
    try:
        response = requests.post(
            f"{st.session_state.backend_url}/api/chat",
            json={
                "message": message,
                "model": model,
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_system_status():
    """Get system status from backend"""
    if not st.session_state.backend_url:
        return {"error": "Backend not available"}
    
    try:
        response = requests.get(f"{st.session_state.backend_url}/api/system/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status check failed: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def process_voice_message(message: str):
    """Process voice message automatically like Google Home"""
    if not message.strip():
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get AI response immediately
    response = make_chat_request(message, "llama3.2:1b")
    
    if "error" not in response:
        response_text = response.get("response", "No response")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "model": "llama3.2:1b",
            "timestamp": datetime.now().isoformat()
        })
        
        # Speak the response immediately
        speak_text(response_text)
    else:
        error_message = f"Error: {response['error']}"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_message,
            "model": "llama3.2:1b",
            "timestamp": datetime.now().isoformat()
        })
        speak_text(error_message)

def speak_text(text: str):
    """Convert text to speech"""
    if not st.session_state.voice_enabled:
        return
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # Clean text for speech
        clean_text = text.replace('**', '').replace('*', '').replace('`', '')
        clean_text = clean_text.replace('#', '').replace('━', '')
        
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "... and more"
        
        engine.say(clean_text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

def display_header():
    """Display main header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🎤 SutazAI RealtimeSTT Chat</h1>
        <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Advanced AI System with Superior Real-time Voice Recognition</p>
    </div>
    """, unsafe_allow_html=True)

def display_realtime_controls():
    """Display RealtimeSTT controls"""
    st.markdown("### 🏠 Google Home Mode - Just Talk!")
    
    # Microphone permission status
    st.markdown("""
    <div id="mic-permission-status" style="text-align: center; padding: 10px; margin: 10px 0; background: #f0f0f0; border-radius: 10px; font-weight: bold;">
        🎤 Checking microphone permission...
    </div>
    
    <script>
    // Immediate permission check and request
    (async function() {
        console.log('Checking microphone permission on page load...');
        try {
            // Check if getUserMedia is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia is not supported in this browser');
            }
            
            // Check current permission state
            if (navigator.permissions) {
                const permission = await navigator.permissions.query({ name: 'microphone' });
                console.log('Current microphone permission state:', permission.state);
                
                if (permission.state === 'granted') {
                    document.getElementById('mic-permission-status').innerHTML = '✅ Microphone Permission: GRANTED';
                    document.getElementById('mic-permission-status').style.color = 'green';
                    return;
                }
            }
            
            // Update status to show checking
            document.getElementById('mic-permission-status').innerHTML = '🔍 Microphone permission required - click button below';
            document.getElementById('mic-permission-status').style.color = 'orange';
            
        } catch (error) {
            console.error('Error checking microphone permission:', error);
            document.getElementById('mic-permission-status').innerHTML = '❌ Error: ' + error.message;
            document.getElementById('mic-permission-status').style.color = 'red';
        }
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # Manual permission request button
    if st.button("🎤 Allow Microphone Access", help="Click to grant microphone permissions", use_container_width=True):
        st.markdown("""
        <script>
        console.log('Button clicked, requesting microphone permission...');
        (async function() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                console.log('Microphone permission granted!', stream);
                
                // Store the stream globally for later use
                window.microphoneStream = stream;
                
                // Update UI
                const statusElement = document.getElementById('mic-permission-status');
                if (statusElement) {
                    statusElement.innerHTML = '✅ Microphone Permission: GRANTED';
                    statusElement.style.color = 'green';
                }
                
                // Show success message
                window.parent.postMessage({type: 'microphone-granted'}, '*');
                
            } catch (error) {
                console.error('Microphone permission denied:', error);
                const statusElement = document.getElementById('mic-permission-status');
                if (statusElement) {
                    statusElement.innerHTML = '❌ Microphone Permission: DENIED - ' + error.message;
                    statusElement.style.color = 'red';
                }
            }
        })();
        </script>
        """, unsafe_allow_html=True)
        st.success("📱 Requesting microphone permission - please allow when prompted!")
    
    # Auto-start recording on first load (after permission check)
    if st.session_state.auto_start and not st.session_state.is_recording:
        start_realtime_recording()
        st.session_state.auto_start = False
    
    # Process any pending transcript updates
    process_transcript_queue()
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.session_state.continuous_mode = st.checkbox(
            "🏠 Google Home Mode", 
            value=st.session_state.continuous_mode,
            help="Continuous conversation like Google Home"
        )
    
    with col2:
        st.session_state.voice_enabled = st.checkbox(
            "🔊 Voice Output", 
            value=st.session_state.voice_enabled,
            help="Enable text-to-speech for AI responses"
        )
    
    with col3:
        if st.session_state.is_recording:
            if st.button("🛑 Stop Listening", use_container_width=True, help="Stop continuous listening"):
                stop_realtime_recording()
                st.session_state.continuous_mode = False
                st.rerun()
        else:
            if st.button("🎤 Start Listening", use_container_width=True, help="Start continuous listening"):
                start_realtime_recording()
                st.rerun()
    
    with col4:
        if st.button("🔄 Reset", use_container_width=True):
            if st.session_state.realtime_stt:
                stop_realtime_recording()
            st.session_state.realtime_stt = None
            st.session_state.is_recording = False
            st.session_state.current_transcript = ""
            st.session_state.continuous_mode = True
            st.session_state.auto_start = True
            st.rerun()
    
    # Google Home style status display
    if st.session_state.continuous_mode:
        status_text = "🎤 LISTENING..." if st.session_state.is_recording else "🏠 READY TO TALK"
        st.markdown(f"""
        <div class="realtime-status">
            {status_text} - Just speak naturally!
        </div>
        """, unsafe_allow_html=True)
        
        transcript_display = st.session_state.current_transcript or "Say something..."
        st.markdown(f"""
        <div class="realtime-transcript">
            <strong>🎙️ You're saying:</strong><br>
            {transcript_display}
        </div>
        """, unsafe_allow_html=True)
    
    # Microphone troubleshooting tips
    with st.expander("🔧 Microphone Troubleshooting"):
        st.markdown("""
        **If microphone isn't working:**
        1. 🔒 **Allow microphone permission** when browser asks
        2. 🔄 **Refresh the page** after granting permission
        3. 🎤 **Check your microphone** is connected and working
        4. 🌐 **Use HTTPS** - some browsers require secure connection
        5. 🔊 **Test with other apps** to verify microphone works
        
        **Supported browsers:**
        - ✅ Chrome (recommended)
        - ✅ Firefox
        - ✅ Safari
        - ✅ Edge
        """)

def display_chat_interface():
    """Main chat interface"""
    st.markdown("## 💬 Chat with SutazAI")
    
    # Backend status
    if st.session_state.backend_url:
        st.success(f"🟢 Connected to backend: {st.session_state.backend_url}")
    else:
        st.error("🔴 Backend service not available")
        if st.button("🔄 Retry Connection"):
            st.session_state.backend_url = get_backend_url()
            st.rerun()
        return
    
    # RealtimeSTT controls
    display_realtime_controls()
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col2:
        model_choice = st.selectbox(
            "AI Model",
            ["llama3.2:1b", "deepseek-coder:7b", "codellama:7b"],
            help="Choose which AI model to use for conversation"
        )
    
    # Chat input
    with st.container():
        st.markdown("### 🗨️ Start Your Conversation")
        
        # Use last message from voice input if available
        default_message = st.session_state.last_message if st.session_state.last_message else ""
        
        # Text input for chat
        user_input = st.text_area(
            "Type your message or use voice:",
            value=default_message,
            height=100,
            placeholder="Ask me anything! I can:\n• Generate code in any language\n• Check system status\n• Manage AI agents\n• Execute system commands\n• Help with programming\n• Answer questions about anything",
            key="chat_input",
            help="Type your message here or use RealtimeSTT voice input"
        )
        
        # Clear the last message after using it
        if st.session_state.last_message:
            st.session_state.last_message = ""
        
        # Enter key hint
        if st.session_state.continuous_mode:
            st.markdown("""
            <div class="enter-hint">
                🏠 <strong>Google Home Mode Active!</strong> Just talk naturally - no buttons needed!
                💡 Your voice will be processed automatically and I'll respond back.
                <br><br>
                🎤 <strong>First time?</strong> Your browser will ask for microphone permission - please click "Allow"!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="enter-hint">
                💡 <strong>Enter Key Enabled!</strong> Press Enter to send, Shift+Enter for new line. 
                🎤 <strong>RealtimeSTT Active!</strong> Real-time voice recognition available.
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            send_button = st.button("🚀 Send Message", use_container_width=True, type="primary")
        
        with col2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("📊 System Status", use_container_width=True):
                status = get_system_status()
                if "error" not in status:
                    st.session_state.chat_history.append({
                        "role": "system",
                        "content": f"System Status Retrieved: {json.dumps(status, indent=2)}",
                        "timestamp": datetime.now().isoformat()
                    })
                    st.rerun()
        
        with col4:
            if st.button("🎤 Test Voice", use_container_width=True):
                if st.session_state.voice_enabled:
                    speak_text("Hello! RealtimeSTT voice system is working perfectly.")
                else:
                    st.info("Enable voice output to test speech synthesis.")
    
    # Process send message
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get AI response
        with st.spinner("🧠 AI is thinking..."):
            response = make_chat_request(user_input, model_choice)
        
        # Add AI response
        if "error" not in response:
            response_text = response.get("response", "No response")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "model": model_choice,
                "timestamp": datetime.now().isoformat()
            })
            
            # Speak the response if voice is enabled
            if st.session_state.voice_enabled:
                speak_text(response_text)
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {response['error']}",
                "model": model_choice,
                "timestamp": datetime.now().isoformat()
            })
        
        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💭 Conversation History")
        
        # Show recent messages (last 20)
        for message in reversed(st.session_state.chat_history[-20:]):
            timestamp = datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
            
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You</strong> <span style="color: #666; font-size: 0.8em;">({timestamp})</span>
                    <br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "assistant":
                model_info = f" • {message.get('model', 'AI')}" if message.get('model') else ""
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 SutazAI</strong> <span style="color: #666; font-size: 0.8em;">({timestamp}{model_info})</span>
                    <br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "system":
                st.markdown(f"""
                <div class="system-message">
                    <strong>⚙️ System</strong> <span style="color: #666; font-size: 0.8em;">({timestamp})</span>
                    <br><br>
                    <pre>{message["content"]}</pre>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.markdown("---")
        if st.session_state.continuous_mode:
            st.info("🏠 Welcome! Google Home Mode is active - just start talking! I'll listen, respond, and keep the conversation going naturally. Make sure to allow microphone access when prompted!")
        else:
            st.info("👋 Welcome! Start a conversation by typing, pressing Enter, or using RealtimeSTT voice input. Your intelligent AI assistant is ready!")

def display_sidebar():
    """Display sidebar with system information"""
    st.sidebar.markdown("# 🎤 RealtimeSTT Control")
    
    # Quick actions
    if st.sidebar.button("🔄 Refresh System"):
        st.session_state.backend_url = get_backend_url()
        st.rerun()
    
    if st.sidebar.button("🧹 Clear All History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # RealtimeSTT Status
    st.sidebar.markdown("## 🎯 Voice Features")
    recording_status = "🔴 RECORDING" if st.session_state.is_recording else "⚪ READY"
    voice_status = "🔊 ON" if st.session_state.voice_enabled else "🔇 OFF"
    realtime_status = "⚡ ACTIVE" if st.session_state.realtime_enabled else "⏸️ PAUSED"
    
    st.sidebar.markdown(f"**Recording:** {recording_status}")
    st.sidebar.markdown(f"**Voice Output:** {voice_status}")
    st.sidebar.markdown(f"**Real-time:** {realtime_status}")
    
    # Microphone permission reminder
    st.sidebar.markdown("## 🎤 Microphone")
    st.sidebar.markdown("**Status:** Check main page")
    st.sidebar.markdown("**Required:** Browser permission")
    if st.sidebar.button("🔄 Request Permission"):
        st.sidebar.info("Click the 'Allow Microphone Access' button on the main page!")
    
    # System status
    st.sidebar.markdown("## 📊 System Status")
    
    if st.session_state.backend_url:
        st.sidebar.success("🟢 Backend: Connected")
        
        # Get and display system info
        status = get_system_status()
        if "error" not in status:
            st.sidebar.text("✅ API: Operational")
            st.sidebar.text("🤖 AI: Ready")
            st.sidebar.text("💾 Database: Active")
            st.sidebar.text("🎤 RealtimeSTT: Available")
            st.sidebar.text("🎙️ Microphone: Check permissions")
        else:
            st.sidebar.warning("⚠️ Some services may be unavailable")
    else:
        st.sidebar.error("🔴 Backend: Disconnected")
    
    # Sample commands
    st.sidebar.markdown("## 💡 Try These Commands")
    
    sample_commands = [
        "What's the system status?",
        "Generate Python code for sorting",
        "Tell me about RealtimeSTT",
        "Check system health",
        "Write a JavaScript function",
        "Show me voice features",
        "How does real-time transcription work?",
        "Test voice input and output"
    ]
    
    for cmd in sample_commands:
        if st.sidebar.button(f"💬 {cmd}", key=f"sample_{cmd}"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": cmd,
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎤 SutazAI RealtimeSTT")
    st.sidebar.markdown("*Superior Voice Recognition*")
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
    st.sidebar.markdown(f"**Backend:** {st.session_state.backend_url or 'Disconnected'}")
    st.sidebar.markdown(f"**Voice Engine:** RealtimeSTT")
    st.sidebar.markdown(f"**Microphone:** Browser permission required")

def main():
    """Main application"""
    display_header()
    display_sidebar()
    display_chat_interface()
    
    # Auto-refresh for real-time updates
    if st.session_state.realtime_enabled or st.session_state.is_recording or st.session_state.continuous_mode:
        time.sleep(0.1)  # Small delay for real-time updates
        st.rerun()

if __name__ == "__main__":
    main()