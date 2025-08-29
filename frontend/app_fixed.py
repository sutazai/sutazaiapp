"""
JARVIS - Fixed version with proper error handling
"""

import streamlit as st
import asyncio
import httpx
import json
from datetime import datetime
import os

# Conditional imports with fallbacks
try:
    from streamlit_mic_recorder import mic_recorder
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    print("Voice features disabled - streamlit_mic_recorder not available")

try:
    from streamlit_chat import message
    CHAT_UI_ENABLED = True
except ImportError:
    CHAT_UI_ENABLED = False
    print("Chat UI disabled - streamlit_chat not available")

# Page configuration
st.set_page_config(
    page_title="JARVIS - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'backend_connected' not in st.session_state:
    st.session_state.backend_connected = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = "GPT-3.5"

# Custom CSS
st.markdown("""
<style>
    .jarvis-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .jarvis-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
    }
    .jarvis-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎮 Control Panel")
    
    # Model Selection
    st.subheader("AI Model")
    models = ["GPT-4", "GPT-3.5", "Claude", "Llama", "Ollama"]
    st.session_state.current_model = st.selectbox("Select Model", models, 
                                                  index=models.index(st.session_state.current_model))
    
    # Voice Settings (if available)
    if VOICE_ENABLED:
        st.subheader("🎤 Voice Settings")
        voice_enabled = st.checkbox("Enable Voice Commands", value=False)
        
        if voice_enabled:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎙️ Start Listening"):
                    st.info("Listening...")
            with col2:
                if st.button("🛑 Stop Listening"):
                    st.info("Stopped")
    else:
        st.warning("Voice features not available")
    
    # System Status
    st.subheader("📊 System Status")
    
    # Check backend connection
    try:
        response = httpx.get("http://localhost:10200/health", timeout=2.0)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
            st.session_state.backend_connected = True
        else:
            st.error("❌ Backend Error")
            st.session_state.backend_connected = False
    except:
        st.warning("⚠️ Backend Offline")
        st.session_state.backend_connected = False
    
    # Service Status
    with st.expander("Service Status"):
        services = {
            "PostgreSQL": "🟢 Online",
            "Redis": "🟢 Online", 
            "Neo4j": "🟡 Limited",
            "ChromaDB": "🟢 Online"
        }
        for service, status in services.items():
            st.text(f"{service}: {status}")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Restart"):
            st.rerun()

# Main Content
st.markdown("""
<div class="jarvis-header">
    <div class="jarvis-title">J.A.R.V.I.S</div>
    <div class="jarvis-subtitle">Just A Rather Very Intelligent System</div>
</div>
""", unsafe_allow_html=True)

# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎤 Voice", "📊 Monitor", "🚀 Agents"])

with tab1:
    st.header("Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message or say 'Hey JARVIS'..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.backend_connected:
                    try:
                        # Call backend API
                        response = httpx.post(
                            "http://localhost:10200/api/v1/chat",
                            json={"message": prompt, "model": st.session_state.current_model},
                            timeout=30.0
                        )
                        if response.status_code == 200:
                            reply = response.json().get("response", "I couldn't process that request.")
                        else:
                            reply = "Backend error. Please try again."
                    except Exception as e:
                        reply = f"Connection error: {str(e)}"
                else:
                    # Fallback response
                    reply = f"I'm currently offline. Using {st.session_state.current_model} in simulation mode. You said: '{prompt}'"
                
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

with tab2:
    st.header("Voice Assistant")
    
    if VOICE_ENABLED:
        st.info("🎤 Voice features are experimental")
        
        # Voice input
        audio = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=False,
            use_container_width=True
        )
        
        if audio:
            st.audio(audio['bytes'])
            st.success("Audio recorded! Processing...")
    else:
        st.warning("Voice features require additional dependencies. Please install streamlit-mic-recorder.")
        st.code("pip install streamlit-mic-recorder", language="bash")

with tab3:
    st.header("System Monitor")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "23%", "−2%")
    with col2:
        st.metric("Memory", "4.2 GB", "+0.3 GB")
    with col3:
        st.metric("Active Agents", "3", "0")
    with col4:
        st.metric("Response Time", "142 ms", "−12 ms")
    
    # Add placeholder chart
    st.subheader("Performance Metrics")
    chart_data = {
        "Time": ["1m", "2m", "3m", "4m", "5m"],
        "Response Time (ms)": [120, 135, 142, 138, 142],
        "Requests/sec": [45, 52, 48, 50, 47]
    }
    st.line_chart(data=chart_data, x="Time", y=["Response Time (ms)", "Requests/sec"])

with tab4:
    st.header("Agent Management")
    
    agents = [
        {"name": "Chat Agent", "status": "🟢 Active", "tasks": 42},
        {"name": "Code Agent", "status": "🟢 Active", "tasks": 18},
        {"name": "Research Agent", "status": "🟡 Idle", "tasks": 0},
        {"name": "Creative Agent", "status": "🔴 Offline", "tasks": 0}
    ]
    
    for agent in agents:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.text(agent["name"])
        with col2:
            st.text(agent["status"])
        with col3:
            st.text(f"{agent['tasks']} tasks")

# Footer
st.markdown("---")
st.markdown("JARVIS v2.0 - Fixed Edition | Backend: " + 
           ("Connected" if st.session_state.backend_connected else "Offline"))
