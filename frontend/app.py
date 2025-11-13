"""
JARVIS - SutazAI Advanced Voice Assistant
Fully integrated frontend with backend connectivity
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
from streamlit_lottie import st_lottie
import time
from datetime import datetime
import json
import base64
import io
import requests
import threading
import plotly.graph_objects as go
import numpy as np

# Try to import webrtc if available
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    WEBRTC_AVAILABLE = True
    
    # Define AudioProcessor class if webrtc is available
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.audio_buffer = []
        
        def recv(self, frame):
            # Store audio frames
            self.audio_buffer.append(frame.to_ndarray())
            return frame
        
        def get_audio_bytes(self):
            if self.audio_buffer:
                import numpy as np
                audio_data = np.concatenate(self.audio_buffer)
                return audio_data.tobytes()
            return None
except ImportError:
    WEBRTC_AVAILABLE = False

# Custom imports
from config.settings import settings
from components.chat_interface import ChatInterface
from components.voice_assistant import VoiceAssistant
from components.system_monitor import SystemMonitor
from services.backend_client_fixed import BackendClient
from services.agent_orchestrator import AgentOrchestrator

# Page configuration
st.set_page_config(
    page_title="JARVIS - AI Assistant",
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
    
    /* Connection status indicator */
    .connection-status {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        z-index: 1000;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-connected {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.5);
    }
    
    .status-disconnected {
        background: linear-gradient(135deg, #F44336 0%, #da190b 100%);
        color: white;
        box-shadow: 0 0 20px rgba(244, 67, 54, 0.5);
    }
    
    /* JARVIS Arc Reactor */
    .arc-reactor {
        width: 100px;
        height: 100px;
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
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 20px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        margin: 20px 0;
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
    
    /* Model selector */
    .model-selector {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00D4FF;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Agent cards */
    .agent-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* WebSocket status */
    .ws-status {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .ws-connected {
        background: #4CAF50;
        animation: pulse 2s infinite;
    }
    
    .ws-disconnected {
        background: #F44336;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.backend_client = BackendClient(settings.BACKEND_URL)
    st.session_state.chat_interface = ChatInterface()
    st.session_state.voice_assistant = VoiceAssistant()
    st.session_state.agent_orchestrator = AgentOrchestrator()
    st.session_state.current_model = "tinyllama:latest"
    st.session_state.current_agent = "default"
    st.session_state.backend_connected = False
    st.session_state.websocket_connected = False
    st.session_state.available_models = []
    st.session_state.available_agents = []
    st.session_state.is_listening = False
    st.session_state.is_processing = False

# Function to check backend connection
def check_backend_connection():
    """Check if backend is connected and update status"""
    try:
        health = st.session_state.backend_client.check_health_sync()
        st.session_state.backend_connected = health.get("status") != "error"
        return st.session_state.backend_connected
    except:
        st.session_state.backend_connected = False
        return False

# Function to initialize WebSocket connection
def initialize_websocket():
    """Initialize WebSocket connection for real-time updates"""
    if not st.session_state.websocket_connected:
        def on_ws_message(message):
            """Handle WebSocket messages"""
            if message.get("type") == "chat_update":
                # Update chat in real-time
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "timestamp": datetime.now().isoformat()
                })
            elif message.get("type") == "status_update":
                # Update status
                st.session_state.backend_connected = message.get("connected", False)
        
        def on_ws_error(error):
            """Handle WebSocket errors"""
            st.session_state.websocket_connected = False
            print(f"WebSocket error: {error}")
        
        # Connect WebSocket
        ws_thread = st.session_state.backend_client.connect_websocket(
            on_message=on_ws_message,
            on_error=on_ws_error
        )
        
        if ws_thread and ws_thread.is_alive():
            st.session_state.websocket_connected = True

# Function to load available models and agents
def load_backend_resources():
    """Load available models and agents from backend"""
    if st.session_state.backend_connected:
        try:
            st.session_state.available_models = st.session_state.backend_client.get_models_sync()
            st.session_state.available_agents = st.session_state.backend_client.get_agents_sync()
        except Exception as e:
            print(f"Failed to load resources: {e}")

# Function to process chat message
def process_chat_message(message: str):
    """Process a chat message and get response from backend"""
    st.session_state.is_processing = True
    
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get response from backend
    try:
        response = st.session_state.backend_client.chat_sync(
            message=message,
            agent=st.session_state.current_agent
        )
        
        # Extract response content
        if response.get("success"):
            response_text = response.get("response", "I'm sorry, I didn't understand that.")
        else:
            response_text = response.get("response", "I encountered an error processing your request.")
        
        # Add assistant response to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": response.get("metadata", {})
        })
        
        # Speak response if voice is enabled
        if st.session_state.get("voice_enabled", False):
            synthesize_speech(response_text)
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I encountered an error: {error_message}",
            "timestamp": datetime.now().isoformat()
        })
    
    st.session_state.is_processing = False

# Function to process voice input
def process_voice_input(audio_bytes):
    """Process voice input and convert to text"""
    try:
        # Send audio to backend for processing
        result = st.session_state.backend_client.send_voice_sync(audio_bytes)
        
        if result and "text" in result:
            return result["text"]
        else:
            # Fallback to local processing
            text = st.session_state.voice_assistant.process_audio_bytes(audio_bytes)
            return text
    except Exception as e:
        print(f"Voice processing error: {e}")
        return None

# Main app header with connection status
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="arc-reactor"></div>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #00D4FF;'>J.A.R.V.I.S</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #999;'>Just A Rather Very Intelligent System</p>", unsafe_allow_html=True)

# Connection status indicator
backend_status = check_backend_connection()
status_class = "status-connected" if backend_status else "status-disconnected"
status_text = "Connected" if backend_status else "Disconnected"
st.markdown(
    f'<div class="connection-status {status_class}">Backend: {status_text}</div>',
    unsafe_allow_html=True
)

# Initialize resources if connected
if backend_status:
    load_backend_resources()
    initialize_websocket()

# Sidebar with controls
with st.sidebar:
    st.markdown("## 🎮 Control Panel")
    
    # Model selector
    st.markdown("### 🤖 AI Model")
    if st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Model",
            st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.current_model) 
                if st.session_state.current_model in st.session_state.available_models else 0,
            key="model_selector"
        )
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            st.success(f"Switched to {selected_model}")
    else:
        st.info("No models available. Using default.")
    
    # Agent selector
    st.markdown("### 🚀 AI Agent")
    if st.session_state.available_agents:
        agent_names = [agent["name"] for agent in st.session_state.available_agents]
        agent_ids = [agent["id"] for agent in st.session_state.available_agents]
        
        current_idx = 0
        if st.session_state.current_agent in agent_ids:
            current_idx = agent_ids.index(st.session_state.current_agent)
        
        selected_agent_idx = st.selectbox(
            "Select Agent",
            range(len(agent_names)),
            format_func=lambda x: agent_names[x],
            index=current_idx,
            key="agent_selector"
        )
        
        selected_agent = agent_ids[selected_agent_idx]
        if selected_agent != st.session_state.current_agent:
            st.session_state.current_agent = selected_agent
            st.success(f"Switched to {agent_names[selected_agent_idx]}")
            
        # Show agent description
        st.caption(st.session_state.available_agents[selected_agent_idx].get("description", ""))
    else:
        st.info("Using default agent")
    
    # Voice settings
    st.markdown("### 🎤 Voice Settings")
    st.session_state.voice_enabled = st.toggle("Enable Voice", value=False)
    
    if st.session_state.voice_enabled:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎙️ Start Listening", use_container_width=True):
                st.session_state.is_listening = True
                st.session_state.voice_assistant.start_listening()
        with col2:
            if st.button("🛑 Stop Listening", use_container_width=True):
                st.session_state.is_listening = False
                st.session_state.voice_assistant.stop_listening()
    
    # System status
    st.markdown("### 📊 System Status")
    
    # Backend connection status
    if backend_status:
        st.success("✅ Backend Connected")
        
        # WebSocket status
        ws_status = "connected" if st.session_state.websocket_connected else "disconnected"
        ws_class = "ws-connected" if st.session_state.websocket_connected else "ws-disconnected"
        st.markdown(
            f'<div><span class="ws-status {ws_class}"></span>WebSocket: {ws_status}</div>',
            unsafe_allow_html=True
        )
        
        # Get detailed health status
        health = st.session_state.backend_client.check_health_sync()
        if "services" in health:
            with st.expander("Service Status"):
                for service, status in health["services"].items():
                    if status:
                        st.markdown(f"✅ {service.title()}")
                    else:
                        st.markdown(f"❌ {service.title()}")
    else:
        st.error("❌ Backend Disconnected")
        if st.button("🔄 Retry Connection"):
            if check_backend_connection():
                st.success("Reconnected!")
                st.rerun()
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_export = st.session_state.chat_interface.export_chat()
            st.download_button(
                label="Download",
                data=chat_export,
                file_name=f"jarvis_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎤 Voice", "📊 Monitor", "🚀 Agents"])

with tab1:
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display connection warning if disconnected
    if not backend_status:
        st.warning("⚠️ Backend is disconnected. Responses will be limited.")
    
    # Chat history container
    chat_container = st.container()
    with chat_container:
        if st.session_state.messages:
            # Display all messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                        if "timestamp" in message:
                            st.caption(f"Sent at {message['timestamp']}")
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        if "timestamp" in message:
                            st.caption(f"Replied at {message['timestamp']}")
                        if "metadata" in message and "model" in message["metadata"]:
                            st.caption(f"Model: {message['metadata']['model']}")
        else:
            st.info("Start a conversation by typing a message below or using voice commands.")
    
    # Processing indicator
    if st.session_state.is_processing:
        with st.spinner("JARVIS is thinking..."):
            time.sleep(0.5)  # Brief pause for visual feedback
    
    # Chat input
    user_input = st.chat_input("Type your message or say 'Hey JARVIS'...")
    if user_input:
        process_chat_message(user_input)
        st.rerun()

with tab2:
    # Voice command center
    st.markdown("### 🎙️ Voice Command Center")
    
    # Check voice service status
    voice_status = st.session_state.backend_client.check_voice_status_sync()
    if voice_status.get("status") == "ready":
        st.success("🎤 Voice service is ready")
    else:
        st.warning(f"⚠️ Voice service status: {voice_status.get('message', 'Unknown')}")
    
    # Simple audio recording section
    st.markdown("### 🎙️ Voice Recording")
    
    # File upload method for audio
    st.markdown("Upload an audio file to transcribe:")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "ogg", "m4a"],
        key="audio_uploader"
    )
    
    if uploaded_file is not None:
        # Display the uploaded audio
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Transcribe Audio", use_container_width=True):
                # Read the file bytes
                audio_bytes = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                # Process the audio
                with st.spinner("Transcribing audio..."):
                    text = process_voice_input(audio_bytes)
                    
                    if text:
                        st.success(f"📝 Transcription: **{text}**")
                        st.session_state.last_transcription = text
                    else:
                        st.error("❌ Could not transcribe audio. Please try a different file.")
        
        with col2:
            if st.button("💬 Send to Chat", use_container_width=True, disabled=not st.session_state.get('last_transcription')):
                if st.session_state.get('last_transcription'):
                    process_chat_message(st.session_state.last_transcription)
                    st.rerun()
    
    # WebRTC recording if available
    if WEBRTC_AVAILABLE:
        st.markdown("### 🎤 Live Recording (Experimental)")
        st.info("Click Start to begin recording from your microphone")
        
        ctx = webrtc_streamer(
            key="voice-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        if ctx.audio_processor:
            if st.button("📤 Process Recording", use_container_width=True):
                audio_bytes = ctx.audio_processor.get_audio_bytes()
                if audio_bytes:
                    # Process the audio
                    with st.spinner("Processing your voice..."):
                        text = process_voice_input(audio_bytes)
                        
                        if text:
                            st.success(f"📝 Transcription: **{text}**")
                            # Process as chat
                            with st.spinner("JARVIS is thinking..."):
                                process_chat_message(text)
                                st.rerun()
                        else:
                            st.error("❌ Could not transcribe audio.")
                else:
                    st.warning("No audio recorded. Please speak into your microphone.")
    
    # Voice commands list
    with st.expander("📝 Available Voice Commands"):
        st.markdown("""
        - **"Hey JARVIS"** - Wake word to activate
        - **"What's the time?"** - Get current time
        - **"What's the weather?"** - Weather information
        - **"Tell me a joke"** - Hear a joke
        - **"Search for [query]"** - Web search
        - **"Analyze [topic]"** - Detailed analysis
        - **"Show system status"** - System metrics
        - **"Switch to [agent]"** - Change AI agent
        - **"Help"** - Show available commands
        """)
    
    # Voice settings and testing
    st.markdown("### ⚙️ Voice Testing")
    
    # Test voice pipeline
    if st.button("🧪 Test Voice Health", use_container_width=True):
        with st.spinner("Checking voice service health..."):
            try:
                health_status = st.session_state.backend_client.check_voice_status_sync()
                if health_status.get("status") == "ready":
                    st.success("✅ Voice service is healthy!")
                    with st.expander("Health Details"):
                        if "details" in health_status:
                            st.json(health_status["details"])
                        else:
                            st.json(health_status)
                elif health_status.get("status") == "degraded":
                    st.warning("⚠️ Voice service is degraded")
                    with st.expander("Health Details"):
                        st.json(health_status.get("details", health_status))
                else:
                    st.error(f"Voice service error: {health_status.get('message', 'Unknown')}")
            except Exception as e:
                st.error(f"Voice health check error: {e}")
    
    # Text to Speech Test
    st.markdown("#### 🔊 Text-to-Speech Test")
    tts_text = st.text_input("Enter text to synthesize:", "Hello, I am JARVIS, your AI assistant.")
    if st.button("🎵 Synthesize Speech", use_container_width=True):
        if tts_text:
            with st.spinner("Synthesizing speech..."):
                success = synthesize_speech(tts_text)
                if success:
                    st.success("✅ Speech synthesized successfully!")
                else:
                    st.error("❌ Speech synthesis failed")

with tab3:
    # System monitoring dashboard
    st.markdown("### 📊 System Monitoring Dashboard")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = SystemMonitor.get_cpu_usage()
        st.metric(
            "CPU Usage", 
            f"{cpu_usage}%",
            delta=f"{cpu_usage-50:.1f}%" if cpu_usage != 0 else None
        )
    
    with col2:
        memory_usage = SystemMonitor.get_memory_usage()
        st.metric(
            "Memory", 
            f"{memory_usage}%",
            delta=f"{memory_usage-50:.1f}%" if memory_usage != 0 else None
        )
    
    with col3:
        disk_usage = SystemMonitor.get_disk_usage()
        st.metric(
            "Disk", 
            f"{disk_usage}%",
            delta=f"{disk_usage-50:.1f}%" if disk_usage != 0 else None
        )
    
    with col4:
        network_speed = SystemMonitor.get_network_speed()
        st.metric("Network", f"{network_speed} MB/s")
    
    # Docker container status
    st.markdown("#### 🐳 Docker Containers")
    
    try:
        containers = SystemMonitor.get_docker_stats()
        
        if containers:
            container_data = []
            for container in containers:
                container_data.append({
                    "Name": container["name"],
                    "Status": "🟢 Running" if container["status"] == "running" else "🔴 Stopped",
                    "CPU": f"{container.get('cpu', 0)}%",
                    "Memory": f"{container.get('memory', 0)} MB",
                    "Uptime": container.get("uptime", "N/A")
                })
            
            st.dataframe(container_data, use_container_width=True)
        else:
            st.info("No container data available")
    except Exception as e:
        st.error(f"Failed to get container stats: {e}")
    
    # Real-time performance chart
    st.markdown("#### 📈 Real-time Performance")
    
    # Create performance chart
    fig = go.Figure()
    
    # Generate sample data (replace with real-time data)
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
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Refresh button
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

with tab4:
    # AI Agents Management
    st.markdown("### 🚀 AI Agents Orchestra")
    
    if st.session_state.available_agents:
        # Display agents in a grid
        agent_cols = st.columns(3)
        
        for i, agent in enumerate(st.session_state.available_agents):
            with agent_cols[i % 3]:
                # Agent card
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent.get('name', 'Unknown Agent')}</h4>
                    <p>{agent.get('description', 'No description available')}</p>
                    <p>Status: {'🟢 Active' if agent.get('id') == st.session_state.current_agent else '⚪ Ready'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(
                    f"{'✓ Active' if agent.get('id') == st.session_state.current_agent else 'Activate'}",
                    key=f"activate_{agent.get('id')}",
                    disabled=agent.get('id') == st.session_state.current_agent
                ):
                    st.session_state.current_agent = agent.get('id')
                    st.success(f"{agent.get('name')} activated!")
                    st.rerun()
    else:
        st.info("No agents available. Please check backend connection.")
    
    # Task orchestration
    st.markdown("### 🎯 Multi-Agent Task Orchestration")
    
    task_description = st.text_area(
        "Describe your complex task:",
        height=100,
        placeholder="Example: Analyze this document, summarize key points, and generate a report with visualizations"
    )
    
    # Agent selection for task
    if st.session_state.available_agents:
        selected_agents = st.multiselect(
            "Select agents for this task:",
            [agent["name"] for agent in st.session_state.available_agents],
            default=[st.session_state.available_agents[0]["name"]] if st.session_state.available_agents else []
        )
    
    col1, col2 = st.columns(2)
    with col1:
        task_priority = st.select_slider(
            "Priority",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )
    
    with col2:
        task_timeout = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=600,
            value=60
        )
    
    if st.button("🚀 Execute Multi-Agent Task", use_container_width=True):
        if task_description and backend_status:
            with st.spinner("Orchestrating agents..."):
                try:
                    # Execute task through agent orchestrator
                    result = st.session_state.agent_orchestrator.execute_task(
                        task_description,
                        agents=selected_agents if 'selected_agents' in locals() else None,
                        priority=task_priority,
                        timeout=task_timeout
                    )
                    
                    st.success("Task execution completed!")
                    
                    # Display results
                    with st.expander("Task Results", expanded=True):
                        if isinstance(result, dict):
                            for key, value in result.items():
                                st.write(f"**{key}:** {value}")
                        else:
                            st.write(result)
                            
                except Exception as e:
                    st.error(f"Task execution failed: {e}")
        elif not task_description:
            st.warning("Please describe the task first")
        else:
            st.error("Backend is not connected")

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: #666;'>JARVIS v2.0 | "
    f"Powered by SutazAI Platform | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)

# Auto-refresh for monitoring tab (optional)
# if tab3 and st.session_state.get("auto_refresh", False):
#     time.sleep(5)
#     st.rerun()