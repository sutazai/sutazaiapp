# frontend/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import json

# Assuming these components exist in the specified paths
# If not, I'll need to create them or they will cause errors.
# For now, I will assume they exist and have a 'render' method.
from components.chat_interface import ChatInterface
from components.code_editor import CodeEditor
from components.document_uploader import DocumentUploader
from components.agent_monitor import AgentMonitor
from components.system_metrics import SystemMetrics
from utils.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_client' not in st.session_state:
    # APIClient will use BACKEND_URL env var automatically
    st.session_state.api_client = APIClient()

if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🤖 SutazAI AGI/ASI Autonomous System")
st.markdown("*Enterprise-grade AI system with zero external dependencies*")

# Main navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat Interface",
    "⚙️ Code Generation",
    "📄 Document Processing",
    "🤖 Agent Management",
    "📊 System Monitoring",
    "⚙️ Settings"
])

# Tab 1: Chat Interface
with tab1:
    st.header("AI Chat Assistant")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chat_interface = ChatInterface(st.session_state.api_client)
        chat_interface.render()
    
    with col2:
        st.subheader("Chat Options")
        
        model_choice = st.selectbox(
            "Select Model",
            ["deepseek-r1:8b", "qwen3:8b", "llama2", "deepseek-coder:33b"],
            help="Choose the AI model for responses"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Control response creativity"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=8192,
            value=2048,
            step=100,
            help="Maximum response length"
        )
        
        use_rag = st.checkbox(
            "Use RAG",
            value=True,
            help="Enable Retrieval-Augmented Generation"
        )
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Tab 2: Code Generation
with tab2:
    st.header("AI Code Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        code_editor = CodeEditor(st.session_state.api_client)
        code_editor.render()
    
    with col2:
        st.subheader("Code Generation Options")
        
        language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "TypeScript", "Go", "Java", "C++", "Shell", "SQL", "Markdown"],
            help="Select the target programming language"
        )

        st.text_area("Additional Instructions", placeholder="e.g., 'Use functional components in React', 'Add extensive comments'")

        if st.button("Generate Code", key="generate_code"):
            # This would trigger the code generation in the CodeEditor component
            pass

# Tab 3: Document Processing
with tab3:
    st.header("Document Intelligence")
    st.markdown("Upload documents for analysis, summarization, and Q&A.")
    document_uploader = DocumentUploader(st.session_state.api_client)
    document_uploader.render()

# Tab 4: Agent Management
with tab4:
    st.header("Agent Management Dashboard")
    st.markdown("Monitor and manage autonomous AI agents.")
    agent_monitor = AgentMonitor(st.session_state.api_client)
    agent_monitor.render()

# Tab 5: System Monitoring
with tab5:
    st.header("System Performance Monitoring")
    st.markdown("Real-time metrics for system health and performance.")
    system_metrics = SystemMetrics(st.session_state.api_client)
    system_metrics.render()

# Tab 6: Settings
with tab6:
    st.header("System Settings")
    
    st.subheader("API Configuration")
    api_base_url = st.text_input(
        "API Base URL", 
        value=st.session_state.api_client.base_url
    )
    
    if st.button("Update API URL"):
        st.session_state.api_client.base_url = api_base_url
        st.success(f"API Base URL updated to {api_base_url}")
        st.rerun()

    st.divider()

    st.subheader("Theme & Layout")
    theme = st.selectbox("Choose a theme", ["Light", "Dark"])
    # Note: Streamlit doesn't natively support dynamic theme switching after initial load.
    # This is a placeholder for potential future functionality or custom implementation.
    st.warning("Theme switching requires a custom implementation.")

    st.divider()

    st.subheader("Cache Management")
    if st.button("Clear All Caches"):
        try:
            # Assuming the APIClient has a post method
            response = st.session_state.api_client.post("/cache/clear_all")
            if response.status_code == 200:
                st.success("All system caches have been cleared.")
            else:
                st.error(f"Failed to clear caches: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
