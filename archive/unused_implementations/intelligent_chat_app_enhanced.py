#!/usr/bin/env python3
"""
SutazAI Enhanced Intelligent Chat Interface
Real-time intelligent chatbot with voice features and Enter key support
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import asyncio
import websocket
import threading
import speech_recognition as sr
import pyttsx3
import pyaudio
import wave
import io
import base64
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="SutazAI Enhanced Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced chat interface with Enter key support
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
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
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
    .voice-controls {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 20px 0;
    }
    .enter-hint {
        color: #666;
        font-size: 12px;
        margin-top: 5px;
        font-style: italic;
    }
</style>

<script>
// JavaScript for Enter key support
document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        const textareas = document.querySelectorAll('textarea[data-testid="stTextArea"]');
        if (textareas.length > 0 && document.activeElement === textareas[0]) {
            event.preventDefault();
            const sendButton = document.querySelector('button[kind="primary"]');
            if (sendButton) {
                sendButton.click();
            }
        }
    }
});
</script>
""", unsafe_allow_html=True)

# Backend URL - try localhost first, then the IP
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
if 'system_info' not in st.session_state:
    st.session_state.system_info = {}
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""

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

def init_voice_recognition():
    """Initialize voice recognition"""
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        return recognizer, microphone
    except Exception as e:
        st.error(f"Voice recognition initialization failed: {e}")
        return None, None

def init_text_to_speech():
    """Initialize text-to-speech"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        st.error(f"Text-to-speech initialization failed: {e}")
        return None

def listen_for_speech():
    """Listen for speech input"""
    recognizer, microphone = init_voice_recognition()
    if not recognizer or not microphone:
        return None
    
    try:
        with microphone as source:
            st.info("🎤 Listening... Speak now!")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        
        st.info("🧠 Processing speech...")
        text = recognizer.recognize_google(audio)
        st.success(f"✅ Heard: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("⚠️ Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def speak_text(text: str):
    """Convert text to speech"""
    if not st.session_state.voice_enabled:
        return
    
    try:
        engine = init_text_to_speech()
        if engine:
            # Remove markdown formatting for speech
            clean_text = text.replace('**', '').replace('*', '').replace('`', '')
            clean_text = clean_text.replace('#', '').replace('━', '')
            
            # Limit length for speech
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "... and more"
            
            engine.say(clean_text)
            engine.runAndWait()
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

def display_header():
    """Display main header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🤖 SutazAI Enhanced Chat</h1>
        <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Advanced AI System with Voice & Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)

def display_voice_controls():
    """Display voice control interface"""
    st.markdown("### 🎤 Voice Controls")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🎤 Voice Input", use_container_width=True, help="Click to speak your message"):
            spoken_text = listen_for_speech()
            if spoken_text:
                st.session_state.last_message = spoken_text
                st.rerun()
    
    with col2:
        st.session_state.voice_enabled = st.checkbox(
            "🔊 Voice Output", 
            value=st.session_state.voice_enabled,
            help="Enable text-to-speech for AI responses"
        )
    
    with col3:
        if st.button("🎯 Quick Commands", use_container_width=True):
            st.session_state.last_message = "What's the system status?"
            st.rerun()
    
    with col4:
        if st.button("🔄 Reset Voice", use_container_width=True):
            st.session_state.listening = False
            st.session_state.voice_enabled = False
            st.rerun()

def display_chat_interface():
    """Main chat interface with Enter key support"""
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
    
    # Voice controls
    display_voice_controls()
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col2:
        model_choice = st.selectbox(
            "AI Model",
            ["llama3.2:1b", "deepseek-coder:7b", "codellama:7b"],
            help="Choose which AI model to use for conversation"
        )
    
    # Chat input with Enter key support
    with st.container():
        st.markdown("### 🗨️ Start Your Conversation")
        
        # Use last message from voice input if available
        default_message = st.session_state.last_message if st.session_state.last_message else ""
        
        # Text input for chat with Enter key hint
        user_input = st.text_area(
            "Type your message:",
            value=default_message,
            height=100,
            placeholder="Ask me anything! I can:\n• Generate code in any language\n• Check system status\n• Manage AI agents\n• Execute system commands\n• Help with programming\n• Answer questions about anything",
            key="chat_input",
            help="Press Enter to send (Shift+Enter for new line)"
        )
        
        # Clear the last message after using it
        if st.session_state.last_message:
            st.session_state.last_message = ""
        
        # Enter key hint
        st.markdown('<div class="enter-hint">💡 Press Enter to send your message, Shift+Enter for new line</div>', unsafe_allow_html=True)
        
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
            if st.button("🤖 Create Agent", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Create a new AI agent for code assistance",
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()
    
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
        st.info("👋 Welcome! Start a conversation by typing a message above or using voice input. I'm your intelligent AI assistant ready to help with any task!")

def display_sidebar():
    """Display sidebar with system information"""
    st.sidebar.markdown("# 🔧 System Control")
    
    # Quick actions
    if st.sidebar.button("🔄 Refresh System"):
        st.session_state.backend_url = get_backend_url()
        st.rerun()
    
    if st.sidebar.button("🧹 Clear All History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Voice settings
    st.sidebar.markdown("## 🎤 Voice Settings")
    st.sidebar.checkbox("Enable Voice Features", value=True, help="Enable/disable voice input and output")
    
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
            st.sidebar.text("🎤 Voice: Available")
        else:
            st.sidebar.warning("⚠️ Some services may be unavailable")
    else:
        st.sidebar.error("🔴 Backend: Disconnected")
    
    # Sample commands
    st.sidebar.markdown("## 💡 Try These Commands")
    
    sample_commands = [
        "What's the system status?",
        "Generate Python code for sorting",
        "Create an AI agent",
        "Check system health",
        "Write a JavaScript function",
        "Show me system performance",
        "Tell me about voice features",
        "How do I use Enter key?"
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
    st.sidebar.markdown("### 🤖 SutazAI v2.0 Enhanced")
    st.sidebar.markdown("*Intelligent AGI/ASI Platform with Voice*")
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
    st.sidebar.markdown(f"**Backend:** {st.session_state.backend_url or 'Disconnected'}")
    st.sidebar.markdown(f"**Voice:** {'🎤 Enabled' if st.session_state.voice_enabled else '🔇 Disabled'}")

def main():
    """Main application"""
    display_header()
    display_sidebar()
    display_chat_interface()
    
    # Auto-refresh every 30 seconds if no backend
    if not st.session_state.backend_url:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()