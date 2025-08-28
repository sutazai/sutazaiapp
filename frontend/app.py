"""
JARVIS - SutazAI Advanced Voice Assistant
Combines best features from multiple JARVIS implementations:
- Dipeshpal/Jarvis_AI: Server-based processing, extensible functions
- Microsoft/JARVIS: Model chaining, multi-modal capabilities
- Voice recognition, synthesis, and real-time chat
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder, speech_to_text
from streamlit_chat import message
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import asyncio
import httpx
import json
import time
from datetime import datetime
import speech_recognition as sr
import pyttsx3
import threading
import queue
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import base64
import io

# Custom imports
from config.settings import settings
from components.voice_assistant import VoiceAssistant
from components.chat_interface import ChatInterface
from components.system_monitor import SystemMonitor
from services.backend_client import BackendClient
from services.agent_orchestrator import AgentOrchestrator
from utils.audio_processor import AudioProcessor

# Page configuration
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for JARVIS theme
st.markdown("""
<style>
    /* JARVIS Blue Theme */
    :root {
        --jarvis-primary: #00D4FF;
        --jarvis-secondary: #0099CC;
        --jarvis-accent: #FF6B6B;
        --jarvis-dark: #0A0E27;
        --jarvis-light: #E6F3FF;
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);
    }
    
    /* Animated background */
    .jarvis-bg {
        position: fixed;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 50%, rgba(0, 153, 204, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 50% 100%, rgba(255, 107, 107, 0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* JARVIS Arc Reactor */
    .arc-reactor {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, #00D4FF 0%, #0099CC 50%, #004466 100%);
        box-shadow: 
            0 0 60px #00D4FF,
            inset 0 0 30px rgba(0, 212, 255, 0.5);
        animation: reactor-glow 2s ease-in-out infinite;
        margin: auto;
    }
    
    @keyframes reactor-glow {
        0%, 100% { 
            box-shadow: 
                0 0 60px #00D4FF,
                inset 0 0 30px rgba(0, 212, 255, 0.5);
        }
        50% { 
            box-shadow: 
                0 0 100px #00D4FF,
                inset 0 0 50px rgba(0, 212, 255, 0.8);
        }
    }
    
    /* Voice wave animation */
    .voice-wave {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 60px;
        margin: 20px 0;
    }
    
    .voice-wave span {
        width: 4px;
        height: 100%;
        background: var(--jarvis-primary);
        margin: 0 2px;
        animation: wave 1.2s linear infinite;
        border-radius: 20px;
    }
    
    .voice-wave span:nth-child(2) { animation-delay: -1.1s; }
    .voice-wave span:nth-child(3) { animation-delay: -1.0s; }
    .voice-wave span:nth-child(4) { animation-delay: -0.9s; }
    .voice-wave span:nth-child(5) { animation-delay: -0.8s; }
    
    @keyframes wave {
        0%, 40%, 100% {
            transform: scaleY(0.4);
        }
        20% {
            transform: scaleY(1);
        }
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        max-width: 70%;
        float: right;
        clear: both;
    }
    
    .jarvis-message {
        background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 70%;
        float: left;
        clear: both;
    }
    
    /* Holographic effect */
    .hologram {
        position: relative;
        background: linear-gradient(180deg, 
            rgba(0, 212, 255, 0.1) 0%, 
            rgba(0, 212, 255, 0.05) 50%, 
            rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 20px;
        overflow: hidden;
    }
    
    .hologram::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            transparent, #00D4FF, transparent, #0099CC, transparent);
        border-radius: 10px;
        opacity: 0;
        animation: hologram-border 3s linear infinite;
        z-index: -1;
    }
    
    @keyframes hologram-border {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.voice_assistant = VoiceAssistant()
    st.session_state.backend_client = BackendClient(settings.BACKEND_URL)
    st.session_state.agent_orchestrator = AgentOrchestrator()
    st.session_state.listening = False
    st.session_state.speaking = False
    st.session_state.current_agent = "jarvis"
    st.session_state.system_metrics = {}

# Background animation
st.markdown('<div class="jarvis-bg"></div>', unsafe_allow_html=True)

# Header with Arc Reactor
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="arc-reactor"></div>', unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: {settings.PRIMARY_COLOR};'>J.A.R.V.I.S</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #999;'>Just A Rather Very Intelligent System</p>", unsafe_allow_html=True)

# Sidebar with controls
with st.sidebar:
    st.markdown("## 🎮 Control Panel")
    
    # Agent selector
    selected_agent = st.selectbox(
        "Select AI Agent",
        settings.AVAILABLE_AGENTS,
        index=0,
        key="agent_selector"
    )
    
    if selected_agent != st.session_state.current_agent:
        st.session_state.current_agent = selected_agent
        st.success(f"Switched to {selected_agent.upper()}")
    
    # Voice settings
    st.markdown("### 🎤 Voice Settings")
    voice_enabled = st.toggle("Enable Voice Commands", value=True)
    
    if voice_enabled:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎙️ Start Listening", use_container_width=True):
                st.session_state.listening = True
        with col2:
            if st.button("🛑 Stop Listening", use_container_width=True):
                st.session_state.listening = False
    
    # Voice parameters
    with st.expander("Voice Parameters"):
        speaking_rate = st.slider("Speaking Rate", 100, 300, settings.SPEAKING_RATE)
        volume = st.slider("Volume", 0, 100, 80)
        st.selectbox("Voice Gender", ["Male", "Female"])
    
    # System status
    st.markdown("### 📊 System Status")
    
    # Backend connection status
    backend_status = asyncio.run(st.session_state.backend_client.check_health())
    if backend_status:
        st.success("✅ Backend Connected")
        with st.expander("Service Status"):
            services = backend_status.get("services", {})
            for service, status in services.items():
                if status:
                    st.markdown(f"✅ {service}")
                else:
                    st.markdown(f"❌ {service}")
    else:
        st.error("❌ Backend Disconnected")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Restart", use_container_width=True):
            st.session_state.initialized = False
            st.rerun()

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎤 Voice", "📊 Monitor", "🚀 Agents"])

with tab1:
    # Chat interface
    chat_container = st.container()
    
    # Display animated voice wave when listening
    if st.session_state.listening:
        st.markdown("""
        <div class="voice-wave">
            <span></span><span></span><span></span>
            <span></span><span></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Voice recorder
        state = speech_to_text(
            language='en',
            start_prompt="🎤 Listening...",
            stop_prompt="🛑 Stop",
            just_once=False,
            use_container_width=True,
            callback=None,
            key="voice_input"
        )
        
        if state and state.text:
            st.session_state.messages.append({"role": "user", "content": state.text})
            # Process with backend
            response = asyncio.run(
                st.session_state.backend_client.chat(state.text, st.session_state.current_agent)
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Speak response if TTS enabled
            if voice_enabled and st.session_state.speaking:
                st.session_state.voice_assistant.speak(response)
    
    # Chat messages display
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="jarvis-message">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message or say 'Hey JARVIS'...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process with backend
        with st.spinner("Processing..."):
            response = asyncio.run(
                st.session_state.backend_client.chat(user_input, st.session_state.current_agent)
            )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with tab2:
    # Voice command center
    st.markdown("### 🎙️ Voice Command Center")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Mic recorder with visual feedback
        audio = mic_recorder(
            start_prompt="🎤 Click to speak",
            stop_prompt="🛑 Stop recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            callback=None,
            key="mic_recorder"
        )
        
        if audio:
            st.audio(audio["bytes"], format="audio/wav")
            
            # Process audio
            with st.spinner("Processing your voice command..."):
                # Convert audio to text
                text = st.session_state.voice_assistant.audio_to_text(audio["bytes"])
                if text:
                    st.success(f"You said: {text}")
                    
                    # Process command
                    response = asyncio.run(
                        st.session_state.backend_client.chat(text, st.session_state.current_agent)
                    )
                    
                    st.info(f"JARVIS: {response}")
                    
                    # Speak response
                    st.session_state.voice_assistant.speak(response)
    
    # Voice commands list
    with st.expander("📝 Available Voice Commands"):
        st.markdown("""
        - **"Hey JARVIS"** - Wake word to activate
        - **"What's the time?"** - Get current time
        - **"Tell me a joke"** - Hear a joke
        - **"Open [app name]"** - Open applications
        - **"Search for [query]"** - Web search
        - **"Analyze [document]"** - Document analysis
        - **"Show system status"** - System metrics
        - **"Switch to [agent]"** - Change AI agent
        - **"Execute [task]"** - Run specific tasks
        """)

with tab3:
    # System monitoring dashboard
    st.markdown("### 📊 System Monitoring Dashboard")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = SystemMonitor.get_cpu_usage()
        st.metric("CPU Usage", f"{cpu_usage}%", delta=f"{cpu_usage-50:.1f}%")
        
    with col2:
        memory_usage = SystemMonitor.get_memory_usage()
        st.metric("Memory", f"{memory_usage}%", delta=f"{memory_usage-50:.1f}%")
        
    with col3:
        disk_usage = SystemMonitor.get_disk_usage()
        st.metric("Disk", f"{disk_usage}%", delta=f"{disk_usage-50:.1f}%")
        
    with col4:
        network_speed = SystemMonitor.get_network_speed()
        st.metric("Network", f"{network_speed} MB/s")
    
    # Docker container status
    st.markdown("#### 🐳 Docker Containers")
    containers = SystemMonitor.get_docker_stats()
    
    container_data = []
    for container in containers:
        container_data.append({
            "Name": container["name"],
            "Status": "🟢 Running" if container["status"] == "running" else "🔴 Stopped",
            "CPU": f"{container['cpu']}%",
            "Memory": f"{container['memory']} MB",
            "Uptime": container["uptime"]
        })
    
    if container_data:
        st.dataframe(container_data, use_container_width=True)
    
    # Real-time performance chart
    st.markdown("#### 📈 Real-time Performance")
    
    # Create performance chart
    fig = go.Figure()
    
    # Sample data (replace with real-time data)
    time_points = list(range(60))
    cpu_data = [50 + np.random.randn() * 10 for _ in range(60)]
    memory_data = [60 + np.random.randn() * 8 for _ in range(60)]
    
    fig.add_trace(go.Scatter(
        x=time_points, y=cpu_data,
        mode='lines', name='CPU',
        line=dict(color='#00D4FF', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, y=memory_data,
        mode='lines', name='Memory',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="System Performance (Last 60 seconds)",
        xaxis_title="Time (s)",
        yaxis_title="Usage (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # AI Agents Management
    st.markdown("### 🚀 AI Agents Orchestra")
    
    # Agent grid
    agent_cols = st.columns(3)
    
    agents_info = {
        "jarvis": {"icon": "🤖", "status": "Active", "description": "Main assistant"},
        "letta": {"icon": "🧠", "status": "Ready", "description": "Memory specialist"},
        "autogpt": {"icon": "🎯", "status": "Ready", "description": "Task automation"},
        "crewai": {"icon": "👥", "status": "Ready", "description": "Team coordination"},
        "baby-agi": {"icon": "🍼", "status": "Ready", "description": "Goal achievement"},
        "gpt-engineer": {"icon": "⚙️", "status": "Pending", "description": "Code generation"}
    }
    
    for i, (agent_name, info) in enumerate(agents_info.items()):
        with agent_cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div class="hologram">
                    <h4>{info['icon']} {agent_name.upper()}</h4>
                    <p>Status: {info['status']}</p>
                    <p style="font-size: 0.9em; color: #999;">{info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Activate", key=f"activate_{agent_name}"):
                    st.session_state.current_agent = agent_name
                    st.success(f"{agent_name.upper()} activated!")
                    st.rerun()
    
    # Task orchestration
    st.markdown("### 🎯 Task Orchestration")
    
    task_input = st.text_area("Describe your complex task:", height=100)
    if st.button("🚀 Execute Multi-Agent Task"):
        if task_input:
            with st.spinner("Orchestrating agents..."):
                # This would call the agent orchestrator
                result = st.session_state.agent_orchestrator.execute_task(task_input)
                st.success("Task completed!")
                st.markdown(f"**Result:** {result}")

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: #666;'>JARVIS v{settings.APP_VERSION} | "
    f"Powered by SutazAI Platform | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)

# Auto-refresh for monitoring (every 5 seconds)
if tab3:
    time.sleep(5)
    st.rerun()