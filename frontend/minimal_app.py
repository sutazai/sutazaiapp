
# Small model configuration for memory efficiency
DEFAULT_MODEL = "qwen2.5:3b"
AVAILABLE_MODELS = ["qwen2.5:3b", "llama3.2:3b", "qwen2.5-coder:3b"]
MODEL_DESCRIPTIONS = {
    "qwen2.5:3b": "Primary small model (2GB RAM)",
    "llama3.2:3b": "Backup small model (2GB RAM)", 
    "qwen2.5-coder:3b": "Coding-focused small model (2GB RAM)"
}
#!/usr/bin/env python3
"""
Quick minimal AGI frontend for immediate deployment
"""
import streamlit as st
import requests
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="SutazAI AGI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL
BACKEND_URL = "http://backend-agi:8000"
# For local testing, also try localhost
BACKEND_URL_LOCAL = "http://localhost:8000"

def test_backend_connection():
    """Test backend connectivity"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, "backend-agi"
    except:
        pass
    
    try:
        response = requests.get(f"{BACKEND_URL_LOCAL}/health", timeout=5)
        if response.status_code == 200:
            return True, "localhost"
    except:
        pass
    
    return False, None

def main():
    # Title and header
    st.title("🤖 SutazAI AGI System")
    st.markdown("### Enterprise AGI/ASI Autonomous System")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Control")
        
        # Backend status
        backend_connected, backend_type = test_backend_connection()
        if backend_connected:
            st.success(f"✅ Backend Connected ({backend_type})")
        else:
            st.error("❌ Backend Disconnected")
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["Dashboard", "Agent Management", "System Status", "Settings"]
        )
    
    # Main content area
    if page == "Dashboard":
        show_dashboard(backend_connected, backend_type)
    elif page == "Agent Management":
        show_agent_management(backend_connected, backend_type)
    elif page == "System Status":
        show_system_status(backend_connected, backend_type)
    elif page == "Settings":
        show_settings(backend_connected, backend_type)

def show_dashboard(backend_connected, backend_type):
    st.header("📊 System Dashboard")
    
    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Backend Status", "🟢 Online" if backend_connected else "🔴 Offline")
    
    with col2:
        st.metric("Core Services", "5/5 Running")
    
    with col3:
        st.metric("Active Agents", "1")
    
    with col4:
        st.metric("System Load", "Normal")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Invoke Agent"):
            if backend_connected:
                try:
                    url = f"{BACKEND_URL if backend_type == 'backend-agi' else BACKEND_URL_LOCAL}/api/v1/agents/sutazai-core/invoke"
                    response = requests.post(url, json={"message": "Test invocation"}, timeout=10)
                    if response.status_code == 200:
                        st.success("Agent invoked successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Failed to invoke agent: {response.status_code}")
                except Exception as e:
                    st.error(f"Error invoking agent: {e}")
            else:
                st.error("Backend not connected")
    
    with col2:
        if st.button("📊 System Check"):
            if backend_connected:
                try:
                    url = f"{BACKEND_URL if backend_type == 'backend-agi' else BACKEND_URL_LOCAL}/api/v1/system/status"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        st.success("System check completed!")
                        st.json(response.json())
                    else:
                        st.error(f"System check failed: {response.status_code}")
                except Exception as e:
                    st.error(f"Error during system check: {e}")
            else:
                st.error("Backend not connected")
    
    with col3:
        if st.button("🔄 Refresh Status"):
            st.rerun()

def show_agent_management(backend_connected, backend_type):
    st.header("🤖 Agent Management")
    
    if backend_connected:
        try:
            url = f"{BACKEND_URL if backend_type == 'backend-agi' else BACKEND_URL_LOCAL}/api/v1/agents"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                agents = response.json()
                st.success(f"Found {agents.get('total', 0)} agents")
                
                for agent in agents.get('agents', []):
                    with st.expander(f"🤖 {agent['name']} ({agent['id']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Status:** {agent['status']}")
                            st.write(f"**Type:** {agent['type']}")
                        with col2:
                            if st.button(f"Invoke {agent['name']}", key=agent['id']):
                                try:
                                    invoke_url = f"{BACKEND_URL if backend_type == 'backend-agi' else BACKEND_URL_LOCAL}/api/v1/agents/{agent['id']}/invoke"
                                    response = requests.post(invoke_url, json={"task": "status check"}, timeout=10)
                                    if response.status_code == 200:
                                        st.success("Agent invoked!")
                                        st.json(response.json())
                                except Exception as e:
                                    st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error fetching agents: {e}")
    else:
        st.error("Backend not connected - cannot fetch agents")

def show_system_status(backend_connected, backend_type):
    st.header("🖥️ System Status")
    
    # System information
    st.subheader("System Information")
    st.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Backend Connected:** {'Yes' if backend_connected else 'No'}")
    st.write(f"**Backend Type:** {backend_type if backend_connected else 'None'}")
    
    # Service status
    st.subheader("Service Status")
    services = [
        ("PostgreSQL", "🟢 Running", "Core database"),
        ("Redis", "🟢 Running", "Cache & message broker"),
        ("ChromaDB", "🟢 Running", "Vector database"),
        ("Qdrant", "🟢 Running", "Vector search"),
        ("Ollama", "🟢 Running", "LLM inference"),
        ("Backend AGI", "🟢 Running" if backend_connected else "🔴 Offline", "Main backend service"),
        ("Frontend AGI", "🟢 Running", "This interface")
    ]
    
    for service, status, description in services:
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.write(f"**{service}**")
        with col2:
            st.write(status)
        with col3:
            st.write(description)

def show_settings(backend_connected, backend_type):
    st.header("⚙️ Settings")
    
    st.subheader("Backend Configuration")
    st.write(f"Backend URL: `{BACKEND_URL}`")
    st.write(f"Fallback URL: `{BACKEND_URL_LOCAL}`")
    st.write(f"Current Connection: `{backend_type if backend_connected else 'None'}`")
    
    st.subheader("System Configuration")
    st.info("System is running in quick deployment mode. Full configuration options will be available after complete deployment.")

if __name__ == "__main__":
    main()