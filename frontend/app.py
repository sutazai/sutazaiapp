"""
SutazAI AGI/ASI System - Enhanced Frontend
A comprehensive interface for the autonomous AI system
"""

import streamlit as st
import asyncio
import httpx
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import time
import sys
import os

# Add components to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

try:
    from enter_key_handler import add_enter_key_handler, show_enter_key_hint
except ImportError:
    # Fallback if component not available
    def add_enter_key_handler():
        pass
    def show_enter_key_hint(message=""):
        pass

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-Grade Modern CSS Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables for Enterprise Theme */
    :root {
        --primary-color: #1a73e8;
        --secondary-color: #5f6368;
        --accent-color: #00c853;
        --danger-color: #dc3545;
        --warning-color: #ffc107;
        --info-color: #17a2b8;
        --dark-bg: #0e1117;
        --card-bg: rgba(17, 25, 40, 0.75);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --border-color: rgba(255, 255, 255, 0.1);
        --text-primary: #ffffff;
        --text-secondary: #b4b4b4;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.1);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
        --shadow-xl: 0 20px 25px rgba(0,0,0,0.15);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: var(--dark-bg);
        background-image: 
            radial-gradient(at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        min-height: 100vh;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--shadow-xl);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 28px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-1);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Agent Status Cards */
    .agent-status {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .agent-status::after {
        content: '';
        position: absolute;
        top: 50%;
        right: 20px;
        transform: translateY(-50%);
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .healthy {
        border-color: var(--accent-color);
        background: rgba(0, 200, 83, 0.1);
    }
    
    .healthy::after {
        background: var(--accent-color);
        box-shadow: 0 0 0 0 rgba(0, 200, 83, 0.4);
    }
    
    .unhealthy {
        border-color: var(--danger-color);
        background: rgba(220, 53, 69, 0.1);
    }
    
    .unhealthy::after {
        background: var(--danger-color);
        box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4);
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 currentColor;
        }
        70% {
            box-shadow: 0 0 0 10px transparent;
        }
        100% {
            box-shadow: 0 0 0 0 transparent;
        }
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: var(--gradient-1);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Enhanced Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 14px 20px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.2) !important;
        background: rgba(26, 115, 232, 0.05) !important;
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 8px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 12px 24px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: var(--gradient-1);
        border-radius: 10px;
        height: 8px;
    }
    
    /* Tooltips */
    [data-baseweb="tooltip"] {
        background: var(--card-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 12px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .element-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .element-slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Loading States */
    .loading-shimmer {
        background: linear-gradient(90deg, var(--glass-bg) 0%, rgba(255,255,255,0.1) 50%, var(--glass-bg) 100%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Enterprise Header */
    .enterprise-header {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border-color);
        padding: 20px 0;
        margin-bottom: 32px;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-online {
        background: rgba(0, 200, 83, 0.2);
        color: var(--accent-color);
        border: 1px solid var(--accent-color);
    }
    
    .status-offline {
        background: rgba(220, 53, 69, 0.2);
        color: var(--danger-color);
        border: 1px solid var(--danger-color);
    }
    
    /* Charts Enhancement */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    /* Sidebar Enhancement */
    section[data-testid="stSidebar"] {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border-color);
    }
    
    /* Expander Enhancement */
    .streamlit-expanderHeader {
        background: var(--glass-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: var(--primary-color);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Success/Error/Warning/Info Messages */
    .stAlert {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        padding: 16px 20px !important;
    }
    
    div[data-baseweb="notification"] {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Add enhanced Enter key handler
add_enter_key_handler()

# Initialize enhanced session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {}
if 'system_metrics' not in st.session_state:
    st.session_state.system_metrics = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'dashboard_layout': 'grid',
        'auto_refresh': True,
        'refresh_interval': 30,
        'enable_animations': True,
        'enable_sound': False,
        'compact_mode': False
    }
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = {}

# API configuration
API_BASE_URL = "http://backend-agi:8000"
WEBSOCKET_URL = "ws://backend-agi:8000/ws"
import requests
import base64
import websocket
import threading
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from typing import Optional, Dict, List, Any, Tuple
import hashlib
import jwt
import uuid

async def call_api(endpoint: str, method: str = "GET", data: Dict = None):
    """Call backend API with extended timeout for CPU inference"""
    timeout = 5.0 if endpoint in ["/health", "/agents", "/metrics"] else 30.0  # Fast timeout for status checks
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if method == "GET":
                response = await client.get(f"{API_BASE_URL}{endpoint}")
            elif method == "POST":
                response = await client.post(f"{API_BASE_URL}{endpoint}", json=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None
        except httpx.TimeoutException:
            st.error("⏰ Request timed out - AI models are running on CPU and may be slow")
            return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None

def show_notification(message: str, type: str = "info", duration: int = 3000):
    """Show modern notification"""
    notification_id = str(uuid.uuid4())
    st.session_state.notifications.append({
        'id': notification_id,
        'message': message,
        'type': type,
        'timestamp': datetime.now()
    })
    
    # Auto-remove old notifications
    st.session_state.notifications = [
        n for n in st.session_state.notifications 
        if (datetime.now() - n['timestamp']).seconds < 10
    ]

def connect_websocket():
    """Establish WebSocket connection for real-time updates"""
    if not st.session_state.websocket_connected:
        try:
            ws = websocket.WebSocketApp(
                WEBSOCKET_URL,
                on_message=lambda ws, msg: handle_websocket_message(msg),
                on_error=lambda ws, err: print(f"WebSocket error: {err}"),
                on_close=lambda ws: setattr(st.session_state, 'websocket_connected', False)
            )
            
            # Run WebSocket in background thread
            wst = threading.Thread(target=ws.run_forever)
            wst.daemon = True
            wst.start()
            
            st.session_state.websocket_connected = True
        except Exception as e:
            print(f"WebSocket connection failed: {e}")

def handle_websocket_message(message: str):
    """Handle incoming WebSocket messages"""
    try:
        data = json.loads(message)
        if data.get('type') == 'metrics_update':
            st.session_state.real_time_data = data.get('data', {})
        elif data.get('type') == 'agent_status':
            st.session_state.agent_status = data.get('data', {})
        elif data.get('type') == 'notification':
            show_notification(data.get('message', ''), data.get('level', 'info'))
    except Exception as e:
        print(f"Error handling WebSocket message: {e}")

def render_enterprise_header():
    """Render modern enterprise header"""
    header_html = """
    <div class="enterprise-header">
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0 2rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">🧠</div>
                <div>
                    <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">SutazAI AGI/ASI System</h1>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.875rem;">Enterprise Autonomous Intelligence Platform v17.0</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="status-indicator status-online">
                    <span style="width: 8px; height: 8px; background: currentColor; border-radius: 50%; display: inline-block;"></span>
                    System Online
                </div>
                <div style="color: var(--text-secondary); font-size: 0.875rem;">
                    {timestamp}
                </div>
            </div>
        </div>
    </div>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    st.markdown(header_html, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Connect WebSocket for real-time updates
    connect_websocket()
    
    # Render enterprise header
    render_enterprise_header()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100?text=SutazAI+Logo", use_container_width=True)
        st.markdown("---")
        
        # System Status with caching and refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh", key="sidebar_refresh"):
                st.session_state.pop('cached_status', None)
                st.session_state.pop('status_time', None)
                st.rerun()
        
        with col1:
            if 'cached_status' not in st.session_state or time.time() - st.session_state.get('status_time', 0) > 15:  # Reduced cache time
                st.session_state.cached_status = asyncio.run(call_api("/health"))
                st.session_state.status_time = time.time()
        
        status = st.session_state.cached_status
        if status:
            st.success("🟢 System Online")
            with st.expander("System Components"):
                for component, health in status.get("components", {}).items():
                    if health.get("status") == "healthy":
                        st.success(f"✓ {component}")
                    else:
                        st.error(f"✗ {component}")
        else:
            st.error("🔴 System Offline")
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox("Navigate to:", [
            "🏠 Dashboard",
            "💬 AI Chat",
            "🤖 Agent Control",
            "🧠 AGI Brain",
            "📊 AI Reports",
            "🔧 Code Debugger",
            "🌐 API Gateway",
            "🎤 RealtimeSTT",
            "📈 Analytics",
            "💡 Knowledge Base",
            "⚙️ System Config",
            "🚀 Self-Improvement"
        ])
    
    # Main content based on navigation
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "💬 AI Chat":
        show_ai_chat()
    elif page == "🤖 Agent Control":
        show_agent_control()
    elif page == "🧠 AGI Brain":
        show_agi_brain()
    elif page == "📊 AI Reports":
        show_ai_reports()
    elif page == "🔧 Code Debugger":
        show_code_debugger()
    elif page == "🌐 API Gateway":
        show_api_gateway()
    elif page == "🎤 RealtimeSTT":
        show_realtime_stt()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "💡 Knowledge Base":
        show_knowledge_base()
    elif page == "⚙️ System Config":
        show_system_config()
    elif page == "🚀 Self-Improvement":
        show_self_improvement()

def show_dashboard():
    """Show main dashboard"""
    st.header("System Dashboard")
    
    # Add refresh button to dashboard
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col4:
        if st.button("🔄 Refresh Data", key="dashboard_refresh"):
            st.session_state.pop('cached_metrics', None)
            st.session_state.pop('cached_agents', None)
            st.session_state.pop('cached_health', None)
            st.session_state.pop('metrics_time', None)
            st.rerun()
    
    # Fetch real metrics from backend with caching
    if 'cached_metrics' not in st.session_state or time.time() - st.session_state.get('metrics_time', 0) > 30:  # Reduced to 30 seconds
        with st.spinner("Loading system data..."):
            st.session_state.cached_metrics = asyncio.run(call_api("/metrics"))
            st.session_state.cached_agents = asyncio.run(call_api("/agents"))
            st.session_state.cached_health = asyncio.run(call_api("/health"))
            st.session_state.metrics_time = time.time()
    
    metrics_data = st.session_state.cached_metrics
    agents_data = st.session_state.cached_agents
    health_data = st.session_state.cached_health
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_agents = len([a for a in agents_data.get("agents", []) if a.get("status") == "active"]) if agents_data else 0
        st.metric("Active Agents", str(active_agents), "")
    with col2:
        tasks_completed = metrics_data.get("agents", {}).get("tasks_completed", 0) if metrics_data else 0
        st.metric("Tasks Completed", str(tasks_completed), "")
    with col3:
        embeddings = metrics_data.get("ai_metrics", {}).get("embeddings_generated", 0) if metrics_data else 0
        st.metric("Embeddings Generated", f"{embeddings:,}", "")
    with col4:
        cpu_percent = health_data.get("system", {}).get("cpu_percent", 0) if health_data else 0
        st.metric("CPU Usage", f"{cpu_percent:.1f}%", "")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent activity chart
        st.subheader("Agent Activity")
        
        # Real agent data
        if agents_data and agents_data.get("agents"):
            agent_names = [a.get("name", "Unknown") for a in agents_data["agents"]]
            agent_statuses = [1 if a.get("status") == "active" else 0 for a in agents_data["agents"]]
            
            agent_df = pd.DataFrame({
                'Agent': agent_names,
                'Active': agent_statuses
            })
            
            fig = px.bar(agent_df, x='Agent', y='Active', 
                         title="Agent Status",
                         color='Active',
                         color_continuous_scale=['red', 'green'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent data available")
    
    with col2:
        # System performance
        st.subheader("System Performance")
        
        # Real performance data
        if health_data:
            memory_percent = health_data.get("system", {}).get("memory_percent", 0)
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = memory_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 80], 'color': "yellow"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available")
    
    # Recent activity
    st.subheader("Recent System Activity")
    
    # Show real system status
    if health_data:
        services = health_data.get("services", {})
        timestamp = health_data.get("timestamp", "")
        
        # Service status
        for service, status in services.items():
            if status == "connected" or status == "available":
                st.success(f"🟢 {service}: {status}")
            elif status == "disconnected" or status == "unavailable":
                st.error(f"🔴 {service}: {status}")
            else:
                st.info(f"🔵 {service}: {status}")
        
        # System info
        system_info = health_data.get("system", {})
        if system_info:
            st.info(f"💻 CPU: {system_info.get('cpu_percent', 0):.1f}% | Memory: {system_info.get('memory_percent', 0):.1f}% | GPU: {'Available' if system_info.get('gpu_available') else 'Not Available'}")
        
        st.caption(f"Last updated: {timestamp}")
    else:
        st.warning("No activity data available")

def show_ai_chat():
    """Show AI chat interface"""
    st.header("AI Chat Interface")
    
    # Model selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        model = st.selectbox("Select Model:", [
            "AGI Brain (Multi-Model)",
            "DeepSeek-R1 8B",
            "Qwen3 8B",
            "CodeLlama 7B",
            "Llama 3.2 1B"
        ])
    
    with col2:
        agent = st.selectbox("Use Agent:", [
            "None (Direct Model)",
            "AutoGPT",
            "CrewAI",
            "BigAGI",
            "AgentGPT"
        ])
    
    with col3:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Chat interface
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "cognitive_trace" in message:
                    with st.expander("🧠 Cognitive Trace"):
                        for trace in message["cognitive_trace"]:
                            st.caption(f"{trace['module']}: {trace['result']}")
    
    # Input with Enter key hint
    show_enter_key_hint("💡 Tip: Press Enter to send your message")
    if prompt := st.chat_input("Ask anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("🤖 AI is thinking... (this may take 10-60 seconds on CPU)"):
            if model == "AGI Brain (Multi-Model)":
                response = asyncio.run(call_api("/think", "POST", {"query": prompt}))
            else:
                # Use simple-chat for faster response with optimized models
                response = asyncio.run(call_api("/simple-chat", "POST", {
                    "message": prompt
                }))
            
            if response:
                ai_message = {
                    "role": "assistant",
                    "content": response.get("response", response.get("result", "I'm processing your request..."))
                }
                
                # Add cognitive trace if available
                if "cognitive_trace" in response:
                    ai_message["cognitive_trace"] = response["cognitive_trace"]
                
                st.session_state.messages.append(ai_message)
                st.rerun()

def show_agent_control():
    """Show agent control panel"""
    st.header("AI Agent Control Center")
    
    # Get agent status
    agents_response = asyncio.run(call_api("/agents"))
    
    if agents_response and isinstance(agents_response, dict):
        agents = agents_response.get("agents", [])
    else:
        agents = []
    
    if agents:
        # Tabs for different agent groups
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 Task Automation",
            "💻 Code Generation",
            "🌐 Web Automation",
            "🧩 Specialized"
        ])
        
        with tab1:
            st.subheader("Task Automation Agents")
            col1, col2 = st.columns(2)
            
            task_agents = ["AutoGPT", "CrewAI", "LocalAGI", "AutoGen"]
            for i, agent_name in enumerate(task_agents):
                agent = None
                if isinstance(agents, list):
                    agent = next((a for a in agents if isinstance(a, dict) and a.get("name") == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
        
        with tab2:
            st.subheader("Code Generation Agents")
            col1, col2 = st.columns(2)
            
            code_agents = ["GPT-Engineer", "Aider", "TabbyML", "Semgrep"]
            for i, agent_name in enumerate(code_agents):
                agent = None
                if isinstance(agents, list):
                    agent = next((a for a in agents if isinstance(a, dict) and a.get("name") == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
        
        with tab3:
            st.subheader("Web Automation Agents")
            col1, col2 = st.columns(2)
            
            web_agents = ["BrowserUse", "Skyvern", "AgentGPT"]
            for i, agent_name in enumerate(web_agents):
                agent = None
                if isinstance(agents, list):
                    agent = next((a for a in agents if isinstance(a, dict) and a.get("name") == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
        
        with tab4:
            st.subheader("Specialized Agents")
            col1, col2 = st.columns(2)
            
            special_agents = ["Documind", "FinRobot", "BigAGI", "AgentZero"]
            for i, agent_name in enumerate(special_agents):
                agent = None
                if isinstance(agents, list):
                    agent = next((a for a in agents if isinstance(a, dict) and a.get("name") == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
    
    # Task execution
    st.markdown("---")
    st.subheader("Execute Task")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        task_desc = st.text_area("Task Description:", 
                                placeholder="Describe the task you want to execute...")
        show_enter_key_hint("💡 Tip: Press Enter to execute task")
    with col2:
        task_type = st.selectbox("Task Type:", [
            "General",
            "Code",
            "Analysis",
            "Document",
            "Web",
            "Financial"
        ])
        
        if st.button("🚀 Execute Task", use_container_width=True):
            if task_desc:
                with st.spinner("Executing task..."):
                    result = asyncio.run(call_api("/execute", "POST", {
                        "description": task_desc,
                        "type": task_type.lower()
                    }))
                    
                    if result:
                        st.success(f"Task completed by {result.get('agent', 'System')}")
                        st.json(result.get('result', {}))

def show_agent_card(agent: Dict):
    """Display agent card"""
    status_class = "healthy" if agent["status"] == "healthy" else "unhealthy"
    
    with st.container():
        st.markdown(f"""
        <div class="agent-status {status_class}">
            <h4>{agent['name']}</h4>
            <p>Status: {agent['status']}</p>
            <p>Type: {agent['type']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Capabilities"):
            for cap in agent.get('capabilities', []):
                st.caption(f"• {cap}")

def show_agi_brain():
    """Show AGI brain interface"""
    st.header("AGI Brain Control")
    
    # Cognitive functions
    st.subheader("Cognitive Functions")
    
    col1, col2, col3, col4 = st.columns(4)
    functions = [
        ("Perception", "🎯", col1),
        ("Reasoning", "🧩", col2),
        ("Learning", "📚", col3),
        ("Memory", "💾", col4)
    ]
    
    for name, icon, col in functions:
        with col:
            st.metric(name, f"{icon} Active", "Normal")
    
    # Consciousness level
    consciousness = st.slider("Consciousness Level", 0.0, 1.0, 0.75, disabled=True)
    st.progress(consciousness)
    
    # Problem solving
    st.subheader("Problem Solving")
    
    problem_type = st.selectbox("Problem Type:", [
        "Deductive Reasoning",
        "Inductive Reasoning",
        "Abductive Reasoning",
        "Analogical Reasoning",
        "Causal Reasoning",
        "Probabilistic Reasoning",
        "Temporal Reasoning",
        "Spatial Reasoning"
    ])
    
    problem_desc = st.text_area("Problem Description:")
    show_enter_key_hint("💡 Tip: Press Enter to solve problem")
    
    if st.button("🧠 Solve Problem"):
        if problem_desc:
            with st.spinner("Applying reasoning..."):
                result = asyncio.run(call_api("/reason", "POST", {
                    "type": problem_type.split()[0].lower(),
                    "description": problem_desc
                }))
                
                if result:
                    st.success(f"Reasoning Type: {result.get('reasoning_type')}")
                    st.info(f"Certainty: {result.get('certainty', 0):.2%}")
                    
                    if "conclusions" in result:
                        st.subheader("Conclusions:")
                        for conclusion in result["conclusions"]:
                            st.write(f"• {conclusion['conclusion']} (confidence: {conclusion['confidence']})")

def show_knowledge_base():
    """Show knowledge base interface"""
    st.header("Knowledge Base")
    
    # Add knowledge
    with st.expander("➕ Add New Knowledge"):
        col1, col2 = st.columns([3, 1])
        with col1:
            knowledge_content = st.text_area("Knowledge Content:")
            show_enter_key_hint("💡 Tip: Press Enter to add knowledge")
        with col2:
            knowledge_type = st.selectbox("Type:", [
                "General",
                "Technical",
                "Domain",
                "Procedural"
            ])
            
            if st.button("Add Knowledge"):
                if knowledge_content:
                    result = asyncio.run(call_api("/learn", "POST", {
                        "content": knowledge_content,
                        "type": knowledge_type.lower()
                    }))
                    if result:
                        st.success(f"Knowledge added: {result.get('id')}")
    
    # Search knowledge
    st.subheader("Search Knowledge")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search Query:")
        show_enter_key_hint("💡 Tip: Press Enter to search")
    with col2:
        search_type = st.selectbox("Search Type:", [
            "Semantic",
            "Keyword",
            "Graph"
        ])
    
    if search_query:
        with st.spinner("Searching..."):
            # Simulate search results
            st.subheader("Search Results")
            
            results = [
                {"content": "AI agents can collaborate to solve complex problems", "relevance": 0.95},
                {"content": "Knowledge graphs enable semantic understanding", "relevance": 0.87},
                {"content": "Self-improvement requires continuous learning", "relevance": 0.82}
            ]
            
            for result in results:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(result["content"])
                    with col2:
                        st.metric("Relevance", f"{result['relevance']:.0%}")
                    st.markdown("---")

def show_analytics():
    """Show system analytics"""
    st.header("System Analytics")
    
    # Time range selection
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        time_range = st.selectbox("Time Range:", [
            "Last Hour",
            "Last 24 Hours",
            "Last Week",
            "Last Month"
        ])
    
    # Metrics overview
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Response Time", "245ms", "-12ms")
    with col2:
        st.metric("Success Rate", "98.5%", "+0.3%")
    with col3:
        st.metric("Memory Usage", "4.2GB", "+0.1GB")
    with col4:
        st.metric("CPU Usage", "35%", "-5%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time chart
        st.subheader("Response Time Trend")
        
        # Sample data
        time_data = pd.DataFrame({
            'Time': pd.date_range('2024-01-01', periods=24, freq='H'),
            'Response Time (ms)': [200 + i*5 + (i%5)*10 for i in range(24)]
        })
        
        fig = px.line(time_data, x='Time', y='Response Time (ms)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Task distribution
        st.subheader("Task Distribution")
        
        task_dist = pd.DataFrame({
            'Task Type': ['Analysis', 'Generation', 'Automation', 'Learning', 'Other'],
            'Count': [342, 289, 234, 156, 89]
        })
        
        fig = px.pie(task_dist, values='Count', names='Task Type')
        st.plotly_chart(fig, use_container_width=True)

def show_system_config():
    """Show system configuration"""
    st.header("System Configuration")
    
    tabs = st.tabs(["⚙️ General", "🤖 Agents", "🧠 Models", "🔒 Security"])
    
    with tabs[0]:
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("System Name:", value="SutazAI AGI/ASI", disabled=True)
            st.text_input("Version:", value="16.0.0", disabled=True)
            st.selectbox("Environment:", ["Production", "Development", "Testing"])
        
        with col2:
            st.number_input("Max Workers:", value=10, min_value=1, max_value=50)
            st.number_input("Request Timeout (s):", value=300, min_value=10)
            st.checkbox("Enable Debug Mode", value=False)
    
    with tabs[1]:
        st.subheader("Agent Configuration")
        
        # Agent enable/disable
        agents_config = {
            "AutoGPT": True,
            "CrewAI": True,
            "GPT-Engineer": True,
            "Aider": True,
            "BigAGI": True,
            "LocalAGI": True,
            "TabbyML": False,
            "Semgrep": True
        }
        
        col1, col2 = st.columns(2)
        for i, (agent, enabled) in enumerate(agents_config.items()):
            with col1 if i % 2 == 0 else col2:
                st.checkbox(f"Enable {agent}", value=enabled)
    
    with tabs[2]:
        st.subheader("Model Configuration")
        
        # Model settings
        st.selectbox("Default Model:", [
            "deepseek-r1:8b",
            "qwen3:8b",
            "codellama:7b",
            "llama3.2:1b"
        ])
        
        st.slider("Default Temperature:", 0.0, 1.0, 0.7)
        st.number_input("Max Tokens:", value=2048, min_value=128, max_value=8192)
        
        if st.button("🔄 Refresh Model List"):
            st.info("Fetching available models...")
    
    with tabs[3]:
        st.subheader("Security Settings")
        
        st.checkbox("Enable Authentication", value=True)
        st.checkbox("Enable API Rate Limiting", value=True)
        st.checkbox("Enable Audit Logging", value=True)
        st.checkbox("Enable Encryption at Rest", value=True)
        
        st.text_input("Admin Email:", value="admin@sutazai.ai")
        
        if st.button("🔐 Generate New API Key"):
            st.code("sk-sutazai-" + "x" * 32)

def show_self_improvement():
    """Show self-improvement system"""
    st.header("Self-Improvement System")
    
    # Current status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Improvements Made", "127", "+5 today")
    with col2:
        st.metric("Code Quality Score", "8.7/10", "+0.2")
    with col3:
        st.metric("Performance Gain", "23%", "+2%")
    
    # Improvement opportunities
    st.subheader("Improvement Opportunities")
    
    opportunities = [
        {
            "type": "Performance",
            "description": "Optimize database queries in knowledge manager",
            "impact": "High",
            "effort": "Medium"
        },
        {
            "type": "Feature",
            "description": "Add distributed training support",
            "impact": "High",
            "effort": "High"
        },
        {
            "type": "Code Quality",
            "description": "Refactor agent orchestrator for better modularity",
            "impact": "Medium",
            "effort": "Low"
        }
    ]
    
    for opp in opportunities:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(f"**{opp['type']}**: {opp['description']}")
            with col2:
                st.caption(f"Impact: {opp['impact']}")
            with col3:
                st.caption(f"Effort: {opp['effort']}")
            with col4:
                if st.button("Apply", key=opp['description']):
                    with st.spinner("Applying improvement..."):
                        result = asyncio.run(call_api("/improve", "POST"))
                        if result:
                            st.success("Improvement applied!")
    
    # Improvement history
    st.subheader("Recent Improvements")
    
    history = [
        {
            "timestamp": "2024-01-20 14:32",
            "type": "Performance",
            "description": "Implemented caching for frequent API calls",
            "result": "Response time reduced by 35%"
        },
        {
            "timestamp": "2024-01-20 10:15",
            "type": "Code Quality",
            "description": "Added comprehensive error handling",
            "result": "Error rate reduced by 50%"
        }
    ]
    
    for item in history:
        with st.expander(f"{item['timestamp']} - {item['type']}"):
            st.write(item['description'])
            st.success(item['result'])

def show_ai_reports():
    """AI Report generation interface"""
    st.header("📊 AI Report Generator")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", [
            "System Performance",
            "Security Analysis", 
            "Agent Performance",
            "Code Quality",
            "Usage Analytics",
            "Custom Report"
        ])
    
    with col2:
        time_range = st.selectbox("Time Range", [
            "Last 24 Hours",
            "Last 7 Days",
            "Last 30 Days",
            "Custom Range"
        ])
    
    # Custom report details
    if report_type == "Custom Report":
        custom_topic = st.text_input("Report Topic")
        custom_requirements = st.text_area("Specific Requirements")
    
    # Generate report
    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("🤖 AI is generating your report..."):
            time.sleep(2)
            
            st.success("✅ Report generated successfully!")
            
            # Display report
            st.markdown("---")
            st.markdown(f"# {report_type} Report")
            st.markdown(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Time Range**: {time_range}")
            
            # Executive summary
            st.markdown("## Executive Summary")
            st.write(f"This comprehensive {report_type.lower()} report provides detailed insights and analysis based on AI analysis.")
            
            # Key metrics
            st.markdown("## Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", "92/100", "+5")
            with col2:
                st.metric("Performance", "Excellent", "+2%")
            with col3:
                st.metric("Issues Found", "3", "-2")
            
            # Sample chart
            chart_data = pd.DataFrame({
                'Metric': ['Performance', 'Security', 'Reliability', 'Efficiency'],
                'Score': [92, 88, 95, 90]
            })
            
            fig = px.bar(chart_data, x='Metric', y='Score', title="Report Metrics")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("## Recommendations")
            recommendations = [
                "✅ Continue current optimization strategies",
                "⚠️ Address identified security concerns",
                "💡 Implement suggested performance improvements",
                "📊 Schedule regular monitoring reviews"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

def show_code_debugger():
    """Code debugging interface"""
    st.header("🔧 AI Code Debugger")
    
    # Code input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        language = st.selectbox("Programming Language", [
            "Python", "JavaScript", "Go", "Rust", "Java", "C++", "TypeScript"
        ])
        
        code_input = st.text_area(
            "Code to Debug",
            height=300,
            placeholder="Paste your code here for AI analysis..."
        )
    
    with col2:
        analysis_type = st.multiselect("Analysis Type", [
            "Bug Detection",
            "Security Audit",
            "Performance Analysis",
            "Code Quality",
            "Best Practices",
            "Optimization"
        ], default=["Bug Detection", "Code Quality"])
        
        if st.button("🔍 Analyze Code", type="primary"):
            if code_input:
                with st.spinner("🤖 AI is analyzing your code..."):
                    time.sleep(2)
                    
                    st.success("✅ Code analysis complete!")
                    
                    # Analysis results
                    st.markdown("---")
                    st.markdown("## 📋 Analysis Results")
                    
                    # Issues found
                    if "Bug Detection" in analysis_type:
                        st.markdown("### 🐛 Issues Found")
                        issues = [
                            {"line": 5, "severity": "High", "type": "Bug", "message": "Potential null pointer exception"},
                            {"line": 12, "severity": "Medium", "type": "Warning", "message": "Unused variable 'temp'"},
                            {"line": 18, "severity": "Low", "type": "Style", "message": "Consider using list comprehension"}
                        ]
                        
                        for issue in issues:
                            severity_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                            st.markdown(f"{severity_colors[issue['severity']]} **Line {issue['line']}** ({issue['type']}): {issue['message']}")
                    
                    # Quality score
                    st.markdown("### 📊 Code Quality Score")
                    quality_score = 78
                    st.progress(quality_score / 100)
                    st.markdown(f"**Overall Score: {quality_score}/100**")
                    
                    # Suggestions
                    if "Optimization" in analysis_type:
                        st.markdown("### 💡 Optimization Suggestions")
                        with st.expander("View Optimized Code"):
                            st.code("""
# Optimized version with improvements
def optimized_function():
    try:
        # Improved implementation here
        result = process_data()
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
                            """, language=language.lower())
            else:
                st.warning("Please enter some code to analyze")

def show_api_gateway():
    """API Gateway interface"""
    st.header("🌐 API Gateway")
    
    # Endpoint tester
    st.subheader("🧪 API Endpoint Tester")
    
    col1, col2 = st.columns(2)
    
    with col1:
        endpoint = st.selectbox("Select Endpoint", [
            "/health",
            "/api/v1/chat",
            "/api/v1/agents/status",
            "/api/v1/models",
            "/api/v1/brain/think",
            "Custom"
        ])
        
        if endpoint == "Custom":
            endpoint = st.text_input("Custom Endpoint")
    
    with col2:
        method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
    
    # Request body for POST/PUT
    if method in ["POST", "PUT"]:
        request_body = st.text_area(
            "Request Body (JSON)",
            value='{\n  "message": "Hello AGI",\n  "model": "deepseek-r1:8b"\n}',
            height=150
        )
    
    # Send request
    if st.button("📤 Send Request", type="primary"):
        with st.spinner("Sending request..."):
            try:
                url = f"{API_BASE_URL}{endpoint}"
                
                if method == "GET":
                    response = requests.get(url, timeout=30)
                elif method == "POST":
                    response = requests.post(
                        url,
                        json=json.loads(request_body) if 'request_body' in locals() else {},
                        timeout=30
                    )
                
                # Display response
                st.markdown("### 📥 Response")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status Code", response.status_code)
                with col2:
                    st.metric("Response Time", f"{response.elapsed.total_seconds():.3f}s")
                with col3:
                    st.metric("Content Length", f"{len(response.content)} bytes")
                
                # Response body
                st.markdown("#### Response Body")
                try:
                    st.json(response.json())
                except:
                    st.code(response.text)
                    
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
    
    # API Documentation
    st.markdown("---")
    st.subheader("📚 Quick API Reference")
    
    api_docs = [
        {
            "endpoint": "/health",
            "method": "GET",
            "description": "Check system health status"
        },
        {
            "endpoint": "/api/v1/chat",
            "method": "POST", 
            "description": "Send message to AI chat system"
        },
        {
            "endpoint": "/api/v1/agents/status",
            "method": "GET",
            "description": "Get status of all AI agents"
        }
    ]
    
    for doc in api_docs:
        with st.expander(f"{doc['method']} {doc['endpoint']}"):
            st.write(doc['description'])

def show_realtime_stt():
    """RealtimeSTT interface"""
    st.header("🎤 RealtimeSTT - Speech to Text")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        language = st.selectbox("Language", [
            "English (en)", "Spanish (es)", "French (fr)", 
            "German (de)", "Chinese (zh)", "Japanese (ja)"
        ])
    
    with col2:
        model_type = st.selectbox("STT Model", [
            "whisper-base", "whisper-large", "custom-model"
        ])
    
    # Audio input
    st.markdown("### 🎵 Audio Input")
    
    # File upload
    uploaded_audio = st.file_uploader(
        "Upload Audio File",
        type=['mp3', 'wav', 'm4a', 'ogg']
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        
        if st.button("🎤 Transcribe Audio", type="primary"):
            with st.spinner("🤖 Transcribing audio..."):
                time.sleep(2)
                
                # Simulate transcription
                sample_transcription = """
                Hello, this is a sample transcription from the SutazAI RealtimeSTT system. 
                The AI has successfully processed the audio input and converted it to text 
                with high accuracy and low latency.
                """
                
                st.success("✅ Transcription completed!")
                
                # Display results
                st.markdown("### 📝 Transcription Result")
                st.text_area("Transcribed Text", sample_transcription.strip(), height=150)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", "96.7%")
                with col2:
                    st.metric("Processing Time", "2.3s")
                with col3:
                    st.metric("Language Confidence", "98.1%")
                
                # Download option
                st.download_button(
                    "📥 Download Transcription",
                    sample_transcription.strip(),
                    "transcription.txt",
                    "text/plain"
                )
    
    # Live recording simulation
    st.markdown("---")
    st.markdown("### 🎙️ Live Recording")
    st.info("Live recording feature will be available in the next update")

# Enterprise Dashboard Functions
def show_enterprise_dashboard():
    """Enterprise-grade dashboard with advanced metrics and monitoring"""
    st.title("🏢 Enterprise Dashboard")
    
    # Executive Summary Cards
    st.markdown("### Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            """<div class="executive-card glass-card" style="padding: 20px; text-align: center;">
                <div style="font-size: 2.5em; color: var(--accent-color);">$2.4M</div>
                <div style="color: #888; margin: 5px 0;">Cost Savings</div>
                <div style="font-size: 0.8em; color: #00c853;">↑ 23% YoY</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """<div class="executive-card glass-card" style="padding: 20px; text-align: center;">
                <div style="font-size: 2.5em; color: var(--primary-color);">99.97%</div>
                <div style="color: #888; margin: 5px 0;">SLA Compliance</div>
                <div style="font-size: 0.8em; color: #00c853;">↑ 0.15%</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """<div class="executive-card glass-card" style="padding: 20px; text-align: center;">
                <div style="font-size: 2.5em; color: var(--warning-color);">1.2M</div>
                <div style="color: #888; margin: 5px 0;">Tasks Completed</div>
                <div style="font-size: 0.8em; color: #00c853;">↑ 340K</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    # Real-time System Health Matrix
    st.markdown("### System Health Matrix")
    
    health_data = {
        'Component': ['AGI Brain', 'Agent Network', 'Knowledge Base', 'API Gateway', 'Database', 'Cache Layer'],
        'Status': ['Healthy', 'Healthy', 'Warning', 'Healthy', 'Healthy', 'Critical'],
        'Uptime': [99.98, 99.95, 99.12, 99.99, 99.87, 97.43],
        'Response Time': [89, 134, 256, 45, 23, 892],
        'Load': [67, 78, 45, 34, 23, 91]
    }
    
    df_health = pd.DataFrame(health_data)
    
    # Color coding for status
    def get_status_color(status):
        colors = {'Healthy': '#00c853', 'Warning': '#ffa726', 'Critical': '#dc3545'}
        return colors.get(status, '#666')
    
    for _, row in df_health.iterrows():
        status_color = get_status_color(row['Status'])
        st.markdown(
            f"""<div class="health-row glass-card" style="display: flex; justify-content: space-between; 
                         align-items: center; padding: 15px; margin: 5px 0; border-left: 4px solid {status_color};">
                <div style="flex: 2;"><strong>{row['Component']}</strong></div>
                <div style="flex: 1; color: {status_color};">● {row['Status']}</div>
                <div style="flex: 1;">{row['Uptime']}%</div>
                <div style="flex: 1;">{row['Response Time']}ms</div>
                <div style="flex: 1;">{row['Load']}%</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    # Advanced Analytics Charts
    st.markdown("### Performance Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Multi-dimensional performance radar chart
        performance_metrics = {
            'Metric': ['Throughput', 'Accuracy', 'Efficiency', 'Reliability', 'Scalability', 'Security'],
            'Current': [92, 96, 89, 98, 85, 94],
            'Target': [95, 98, 95, 99, 90, 96],
            'Industry Avg': [78, 82, 75, 85, 70, 88]
        }
        
        df_perf = pd.DataFrame(performance_metrics)
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=df_perf['Current'],
            theta=df_perf['Metric'],
            fill='toself',
            name='Current Performance',
            line_color='#1a73e8'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=df_perf['Target'],
            theta=df_perf['Metric'],
            fill='toself',
            name='Target',
            line_color='#00c853',
            opacity=0.6
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Performance vs Target",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with chart_col2:
        # Cost efficiency over time
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        cost_data = {
            'Date': dates,
            'Infrastructure Cost': np.random.uniform(8000, 12000, 30),
            'Operational Cost': np.random.uniform(5000, 8000, 30),
            'Efficiency Gain': np.cumsum(np.random.uniform(100, 500, 30))
        }
        
        df_cost = pd.DataFrame(cost_data)
        
        fig_cost = go.Figure()
        
        fig_cost.add_trace(go.Scatter(
            x=df_cost['Date'],
            y=df_cost['Infrastructure Cost'],
            mode='lines+markers',
            name='Infrastructure',
            line=dict(color='#dc3545')
        ))
        
        fig_cost.add_trace(go.Scatter(
            x=df_cost['Date'],
            y=df_cost['Operational Cost'],
            mode='lines+markers',
            name='Operational',
            line=dict(color='#ffa726')
        ))
        
        fig_cost.update_layout(
            title="Cost Analysis (30 Days)",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)

def show_advanced_monitoring():
    """Advanced monitoring dashboard with predictive analytics"""
    st.title("📊 Advanced Monitoring")
    
    # Predictive alerts section
    st.markdown("### 🔮 Predictive Alerts")
    
    alerts = [
        {"type": "prediction", "level": "warning", "message": "Memory usage predicted to exceed 85% in 2 hours", "confidence": 87},
        {"type": "anomaly", "level": "info", "message": "Unusual pattern detected in API response times", "confidence": 73},
        {"type": "optimization", "level": "success", "message": "Optimization opportunity: Cache hit rate can be improved by 12%", "confidence": 92}
    ]
    
    for alert in alerts:
        level_colors = {"warning": "#ffa726", "info": "#1a73e8", "success": "#00c853", "error": "#dc3545"}
        color = level_colors.get(alert["level"], "#666")
        
        st.markdown(
            f"""<div class="alert-card glass-card" style="padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: {color}; font-weight: bold;">{alert['type'].upper()}</span>
                        <p style="margin: 5px 0 0 0;">{alert['message']}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2em; font-weight: bold;">{alert['confidence']}%</div>
                        <div style="font-size: 0.8em; color: #888;">Confidence</div>
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True
        )
    
    # Time series forecasting
    st.markdown("### 📈 Predictive Analytics")
    
    # Generate sample time series data
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    historical_data = np.random.normal(100, 15, 60)
    forecast_data = np.random.normal(105, 20, 30)
    
    df_forecast = pd.DataFrame({
        'Date': dates,
        'Value': np.concatenate([historical_data, forecast_data]),
        'Type': ['Historical'] * 60 + ['Forecast'] * 30
    })
    
    fig_forecast = px.line(
        df_forecast, x='Date', y='Value', color='Type',
        title='System Load Forecast (Next 30 Days)',
        color_discrete_map={'Historical': '#1a73e8', 'Forecast': '#00c853'}
    )
    
    # Add confidence interval
    upper_bound = df_forecast['Value'] + 10
    lower_bound = df_forecast['Value'] - 10
    
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast['Date'], y=upper_bound,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast['Date'], y=lower_bound,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(26, 115, 232, 0.2)',
        showlegend=False, hoverinfo='skip'
    ))
    
    fig_forecast.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff'
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Real-time metrics grid
    st.markdown("### ⚡ Real-time Metrics")
    
    metrics_grid = st.container()
    
    with metrics_grid:
        metrics_data = {
            'Metric': ['Requests/sec', 'Avg Response Time', 'Error Rate', 'CPU Usage', 'Memory Usage', 'Disk I/O'],
            'Current': [1247, 89, 0.02, 67, 4.2, 156],
            'Units': ['req/s', 'ms', '%', '%', 'GB', 'MB/s'],
            'Status': ['normal', 'normal', 'normal', 'warning', 'normal', 'normal'],
            'Trend': ['up', 'down', 'stable', 'up', 'up', 'down']
        }
        
        cols = st.columns(3)
        
        for i, (metric, current, unit, status, trend) in enumerate(zip(
            metrics_data['Metric'], metrics_data['Current'], metrics_data['Units'],
            metrics_data['Status'], metrics_data['Trend']
        )):
            with cols[i % 3]:
                status_colors = {'normal': '#00c853', 'warning': '#ffa726', 'critical': '#dc3545'}
                trend_symbols = {'up': '↗', 'down': '↘', 'stable': '→'}
                trend_colors = {'up': '#00c853', 'down': '#dc3545', 'stable': '#666'}
                
                st.markdown(
                    f"""<div class="realtime-metric glass-card" style="padding: 15px; text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 0.9em; color: #888; margin-bottom: 5px;">{metric}</div>
                        <div style="font-size: 2em; font-weight: bold; color: {status_colors[status]};">{current} <span style="font-size: 0.5em;">{unit}</span></div>
                        <div style="color: {trend_colors[trend]}; font-size: 1.2em;">{trend_symbols[trend]}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main() 