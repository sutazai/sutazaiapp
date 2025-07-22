"""
SutazAI Frontend - Main Streamlit Application
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="SutazAI AGI/ASI System v9",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #1e2329;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #2d3339;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00ff88;
    }
    .error-message {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #00ff88;
        color: #0e1117;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_health() -> Dict[str, Any]:
    """Check backend health status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Backend returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to backend"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 SutazAI AGI/ASI System v9</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🤖 SutazAI Control")
        page = st.selectbox(
            "Page",
            ["Dashboard", "Chat Interface", "Model Management", "Agent Control", "System Monitor", "Settings"]
        )
        
        # Backend status in sidebar
        health = check_backend_health()
        if health.get("status") == "healthy":
            st.success("✅ Backend connected successfully!")
        else:
            st.error(f"❌ Backend error: {health.get('message', 'Unknown error')}")
        
        # System info
        st.markdown("---")
        st.markdown(f"**Version:** v9.0.0")
        st.markdown(f"**Environment:** {os.getenv('ENVIRONMENT', 'development')}")
        st.markdown("Powered by Streamlit")
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard(health)
    elif page == "Chat Interface":
        show_chat_interface()
    elif page == "Model Management":
        show_model_management()
    elif page == "Agent Control":
        show_agent_control()
    elif page == "System Monitor":
        show_system_monitor()
    elif page == "Settings":
        show_settings()

def show_dashboard(health: Dict[str, Any]):
    """Display main dashboard"""
    st.markdown("## System Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Status**")
        status = "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Error"
        st.markdown(f'<div class="metric-value">{status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Agents**")
        agent_count = health.get("services", {}).get("agents", {}).get("active_count", 0)
        st.markdown(f'<div class="metric-value">{agent_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Models**")
        model_count = health.get("services", {}).get("models", {}).get("loaded_count", 0)
        st.markdown(f'<div class="metric-value">{model_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Requests**")
        st.markdown(f'<div class="metric-value">0</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Services status
    st.markdown("---")
    st.markdown("### Services")
    
    if health.get("status") == "healthy":
        services = health.get("services", {})
        
        service_cols = st.columns(len(services))
        for idx, (service, status) in enumerate(services.items()):
            with service_cols[idx]:
                st.markdown(f"**{service.title()}**")
                if isinstance(status, dict):
                    st.success(f"✅ {status.get('status', 'connected')}")
                else:
                    st.success(f"✅ {status}")
    else:
        st.error("⚠️ Services unavailable - Backend not connected")
    
    # GPU Status
    st.markdown("---")
    st.markdown("### GPU Status")
    gpu_available = health.get("gpu_available", False)
    if gpu_available:
        st.success("🎮 GPU detected and available")
    else:
        st.info("💻 Running in CPU mode")

def show_chat_interface():
    """Display chat interface"""
    st.markdown("## Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask SutazAI anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/api/v1/chat",
                        json={"message": prompt},
                        timeout=API_TIMEOUT
                    )
                    if response.status_code == 200:
                        ai_response = response.json().get("response", "I couldn't generate a response.")
                    else:
                        ai_response = "Sorry, I encountered an error. Please check the backend connection."
                except:
                    ai_response = "Sorry, I couldn't connect to the backend service."
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

def show_model_management():
    """Display model management interface"""
    st.markdown("## Model Management")
    
    try:
        # Get available models
        response = requests.get(f"{BACKEND_URL}/api/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            
            if models:
                st.markdown("### Available Models")
                for model in models:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(model.get("name", "Unknown"))
                    with col2:
                        st.text(model.get("size", "N/A"))
                    with col3:
                        if st.button("Load", key=f"load_{model.get('name')}"):
                            st.info(f"Loading {model.get('name')}...")
            else:
                st.info("No models available. Please download models first.")
        else:
            st.error("Failed to fetch models from backend")
    except:
        st.error("Cannot connect to backend service")
    
    # Model download section
    st.markdown("---")
    st.markdown("### Download New Model")
    model_name = st.text_input("Model name (e.g., deepseek-r1:8b)")
    if st.button("Download Model"):
        if model_name:
            st.info(f"Downloading {model_name}... This may take several minutes.")
            # TODO: Implement model download

def show_agent_control():
    """Display agent control interface"""
    st.markdown("## Agent Control")
    
    st.info("Agent control interface will be available in the next update.")
    
    # Placeholder for agent list
    st.markdown("### Available Agents")
    agents = ["AutoGPT", "CrewAI", "AgentGPT", "PrivateGPT", "LlamaIndex"]
    
    for agent in agents:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.text(f"🤖 {agent}")
        with col2:
            st.text("⚫ Stopped")
        with col3:
            st.button("Start", key=f"start_{agent}")

def show_system_monitor():
    """Display system monitoring"""
    st.markdown("## System Monitor")
    
    # Create placeholder charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CPU Usage")
        # Placeholder data
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'cpu': [20 + i % 30 for i in range(100)]
        })
        fig = px.line(df, x='time', y='cpu', title='CPU Usage %')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Memory Usage")
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'memory': [40 + i % 20 for i in range(100)]
        })
        fig = px.line(df, x='time', y='memory', title='Memory Usage %')
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Display settings page"""
    st.markdown("## Settings")
    
    # API Settings
    st.markdown("### API Configuration")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    api_key = st.text_input("API Key", type="password", value="")
    
    # Model Settings
    st.markdown("### Model Settings")
    default_model = st.selectbox("Default Model", ["deepseek-r1:8b", "qwen3:8b", "codellama:7b"])
    max_tokens = st.slider("Max Tokens", 100, 4096, 2048)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    
    # Save button
    if st.button("Save Settings"):
        st.success("✅ Settings saved successfully!")

if __name__ == "__main__":
    main()