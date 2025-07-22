"""
SutazAI AGI/ASI System - Modern Intelligent Frontend
Real-time, responsive interface with advanced AI capabilities
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import os

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .status-card.offline {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .status-card.warning {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .agent-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 3px solid #3b82f6;
    }
    
    .agent-card.active {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    .agent-card.inactive {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        height: 400px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
    }
    
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .chat-message.user {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
    }
    
    .chat-message.assistant {
        background: #f0fdf4;
        border-left: 3px solid #10b981;
    }
    
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #10b981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
    }
    
    .status-offline {
        background-color: #ef4444;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.3);
    }
    
    .status-warning {
        background-color: #f59e0b;
        box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "http://backend-agi:8000")
SESSION_TIMEOUT = 30  # seconds

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_status' not in st.session_state:
    st.session_state.system_status = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

def call_api(endpoint: str, method: str = "GET", data: Dict = None, timeout: int = 10) -> Optional[Dict]:
    """Synchronous API call with proper error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectRefused:
        st.error("🔴 Cannot connect to backend service")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔴 Connection error - Backend service may be down")
        return None
    except Exception as e:
        st.error(f"🔴 Unexpected error: {str(e)}")
        return None

def get_system_status() -> Dict:
    """Get comprehensive system status"""
    status = {
        "overall": "offline",
        "backend": "offline",
        "agents": [],
        "models": [],
        "services": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Test backend health
    health_data = call_api("/health", timeout=5)
    if health_data:
        status["overall"] = "online"
        status["backend"] = "online"
        status["services"] = health_data.get("services", {})
        
        # Get agents
        agents_data = call_api("/agents", timeout=5)
        if agents_data:
            status["agents"] = agents_data.get("agents", [])
        
        # Get models
        models_data = call_api("/models", timeout=5)
        if models_data:
            status["models"] = models_data.get("models", [])
    
    return status

def render_header():
    """Render modern header with real-time status"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 SutazAI AGI/ASI System</h1>
        <p>Autonomous General Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

def render_system_status():
    """Render real-time system status"""
    status = get_system_status()
    
    # Overall status indicator
    if status["overall"] == "online":
        st.markdown("""
        <div class="status-card">
            <span class="status-indicator status-online"></span>
            <strong>System Online</strong> - All services operational
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card offline">
            <span class="status-indicator status-offline"></span>
            <strong>System Offline</strong> - Cannot connect to backend
        </div>
        """, unsafe_allow_html=True)
    
    # Service status grid
    if status["overall"] == "online":
        col1, col2, col3, col4 = st.columns(4)
        
        services = status.get("services", {})
        with col1:
            st.metric("Models", len(status["models"]), "Available")
        with col2:
            st.metric("Agents", len(status["agents"]), "Active")
        with col3:
            st.metric("Database", "✅" if services.get("database") == "connected" else "❌")
        with col4:
            st.metric("Vector DB", "✅" if services.get("chromadb") == "connected" else "❌")
    
    return status

def render_chat_interface(system_status):
    """Render intelligent chat interface"""
    st.subheader("💬 AI Assistant")
    
    if system_status["overall"] != "online":
        st.error("🔴 Chat unavailable - System offline")
        return
    
    # Model selection
    models = system_status.get("models", [])
    if models:
        model_names = [m.get("id", "unknown") for m in models]
        selected_model = st.selectbox("Select Model:", model_names, key="chat_model")
    else:
        st.warning("No models available")
        return
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <strong>AI:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Ask anything...", key="chat_input", placeholder="What would you like to know?")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        send_button = st.button("Send", key="send_chat")
    
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show thinking indicator
        with st.spinner("🤖 AI is thinking..."):
            # Call AI API
            response_data = call_api("/simple-chat", "POST", {
                "message": user_input
            }, timeout=30)
            
            if response_data:
                ai_response = response_data.get("response", "Sorry, I couldn't process your request.")
                
                # Add AI response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now().isoformat(),
                    "model": response_data.get("model", "unknown"),
                    "processing_time": response_data.get("processing_time", 0)
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I'm having trouble connecting to my AI systems. Please try again.",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Rerun to update chat display
        st.rerun()

def render_agents_panel(system_status):
    """Render agents management panel"""
    st.subheader("🤖 AI Agents")
    
    agents = system_status.get("agents", [])
    
    if not agents:
        st.info("No agents available")
        return
    
    for agent in agents:
        status_class = "active" if agent.get("status") == "active" else "inactive"
        health_icon = "🟢" if agent.get("health") == "healthy" else "🔴"
        
        st.markdown(f"""
        <div class="agent-card {status_class}">
            <strong>{health_icon} {agent.get('name', 'Unknown Agent')}</strong><br>
            <small>Type: {agent.get('type', 'unknown')} | Status: {agent.get('status', 'unknown')}</small><br>
            <small>{agent.get('description', 'No description')}</small>
        </div>
        """, unsafe_allow_html=True)

def render_models_panel(system_status):
    """Render models management panel"""
    st.subheader("🧠 AI Models")
    
    models = system_status.get("models", [])
    
    if not models:
        st.info("No models loaded")
        return
    
    for model in models:
        size = model.get("size", "Unknown")
        capabilities = model.get("capabilities", [])
        
        st.markdown(f"""
        <div class="agent-card active">
            <strong>🤖 {model.get('name', 'Unknown Model')}</strong><br>
            <small>Size: {size} | Status: {model.get('status', 'unknown')}</small><br>
            <small>Capabilities: {', '.join(capabilities)}</small>
        </div>
        """, unsafe_allow_html=True)

def render_system_dashboard():
    """Render comprehensive system dashboard"""
    st.subheader("📊 System Dashboard")
    
    # Real-time system metrics
    health_data = call_api("/health", timeout=5)
    
    if health_data:
        system_info = health_data.get("system", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>CPU Usage</h3>
                <h2 style="color: #3b82f6;">{:.1f}%</h2>
            </div>
            """.format(system_info.get("cpu_percent", 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Memory Usage</h3>
                <h2 style="color: #10b981;">{:.1f}%</h2>
            </div>
            """.format(system_info.get("memory_percent", 0)), unsafe_allow_html=True)
        
        with col3:
            gpu_status = "Available" if system_info.get("gpu_available") else "Not Available"
            color = "#10b981" if system_info.get("gpu_available") else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <h3>GPU Status</h3>
                <h2 style="color: {color};">{gpu_status}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Activity log
    st.subheader("📝 Recent Activity")
    activity_log = [
        {"time": "2 min ago", "event": "Model response generated", "status": "success"},
        {"time": "5 min ago", "event": "System health check passed", "status": "info"},
        {"time": "8 min ago", "event": "New chat session started", "status": "info"},
        {"time": "12 min ago", "event": "Backend services connected", "status": "success"},
    ]
    
    for activity in activity_log:
        color = "#10b981" if activity["status"] == "success" else "#3b82f6"
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.5rem 0; border-left: 3px solid {color}; background: #f8fafc;">
            <strong>{activity["time"]}</strong>: {activity["event"]}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Render header
    render_header()
    
    # Get system status
    system_status = get_system_status()
    st.session_state.system_status = system_status
    
    # Auto-refresh every 30 seconds
    if time.time() - st.session_state.last_update > 30:
        st.session_state.last_update = time.time()
        st.rerun()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=SutazAI", use_container_width=True)
        
        # Navigation
        page = st.selectbox("Navigate to:", [
            "🏠 Dashboard",
            "💬 AI Chat", 
            "🤖 Agents",
            "🧠 Models",
            "📊 System Monitor"
        ])
        
        # Quick status
        st.markdown("### Quick Status")
        render_system_status()
        
        # Refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.last_update = 0
            st.rerun()
    
    # Main content based on navigation
    if page == "🏠 Dashboard":
        render_system_dashboard()
    elif page == "💬 AI Chat":
        render_chat_interface(system_status)
    elif page == "🤖 Agents":
        render_agents_panel(system_status)
    elif page == "🧠 Models":
        render_models_panel(system_status)
    elif page == "📊 System Monitor":
        st.subheader("📊 System Monitor")
        st.info("Advanced monitoring features coming soon...")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        SutazAI AGI/ASI System v3.0 | Last Updated: {datetime.now().strftime('%H:%M:%S')} | 
        Status: {'🟢 Online' if system_status['overall'] == 'online' else '🔴 Offline'}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()