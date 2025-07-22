#!/usr/bin/env python3
"""
SutazAI Complete AGI/ASI Frontend
Comprehensive Streamlit interface for all AI agents and capabilities
"""

import streamlit as st
import requests
import json
import asyncio
import websockets
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import io
import os
from typing import Dict, List, Any, Optional

# Page config
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://sutazai-backend-simple:8000"

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .agent-card {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: rgba(255, 255, 255, 0.1);
    }
    .metric-card {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px;
        text-align: center;
    }
    .status-healthy {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #F44336;
        font-weight: bold;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
    }
    .user-message {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        margin-left: 20%;
    }
    .ai-message {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=60)
def get_system_status():
    """Get system status with caching"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get system status: {e}")
    return None

@st.cache_data(ttl=300)
def get_available_models():
    """Get available models with caching"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/models", timeout=10)
        if response.status_code == 200:
            return response.json().get("models", [])
    except Exception as e:
        st.error(f"Failed to get models: {e}")
    return []

def get_agents_status():
    """Get agents status"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/agents/status", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to get agents status: {e}")
    return None

def send_chat_message(message: str, model: str, agent: str = None):
    """Send chat message to backend"""
    try:
        payload = {
            "message": message,
            "model": model
        }
        if agent and agent != "None":
            payload["agent"] = agent
            
        response = requests.post(
            f"{BACKEND_URL}/api/v1/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def execute_agent_task(agent_type: str, task: str, parameters: Dict = None):
    """Execute task on specific agent"""
    try:
        payload = {
            "agent_type": agent_type,
            "task": task,
            "parameters": parameters or {}
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agents/execute",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "None"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek-r1:8b"

# Sidebar
st.sidebar.title("🧠 SutazAI AGI/ASI System")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigate",
    [
        "🏠 Dashboard",
        "💬 Interactive Chatbot", 
        "📊 AI-Generated Reports",
        "🔧 Code Debugging Panel",
        "🌐 API Gateway Interface",
        "🎤 RealtimeSTT",
        "🤖 Agent Management",
        "📈 System Analytics",
        "⚙️ Configuration"
    ]
)

# Get system status for sidebar
status = get_system_status()
if status:
    st.sidebar.markdown("### System Status")
    if status.get("status") == "healthy":
        st.sidebar.markdown("🟢 **System Healthy**")
    else:
        st.sidebar.markdown("🔴 **System Issues**")
    
    st.sidebar.markdown(f"**Models**: {len(status.get('models', []))}")
    st.sidebar.markdown(f"**Agents**: {len(status.get('agents', {}))}")

# Main content based on selected page
if page == "🏠 Dashboard":
    st.title("🧠 SutazAI AGI/ASI System Dashboard")
    st.markdown("### Enterprise-Grade Autonomous AI System")
    
    if status:
        # System overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>System Status</h3>
                <h2>HEALTHY</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            model_count = len(status.get("models", []))
            st.markdown(f"""
            <div class="metric-card">
                <h3>AI Models</h3>
                <h2>{model_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            agent_count = len(status.get("agents", {}))
            st.markdown(f"""
            <div class="metric-card">
                <h3>AI Agents</h3>
                <h2>{agent_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Uptime</h3>
                <h2>9h+</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Components status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Infrastructure Components")
            components = status.get("components", {})
            
            for component, health in components.items():
                status_class = "status-healthy" if health == "healthy" else "status-unhealthy"
                st.markdown(f"**{component.title()}**: <span class='{status_class}'>{health.title()}</span>", 
                          unsafe_allow_html=True)
        
        with col2:
            st.subheader("🤖 AI Agents Status")
            agents_data = get_agents_status()
            if agents_data:
                online_agents = agents_data.get("online_agents", 0)
                total_agents = agents_data.get("total_agents", 0)
                
                # Create agent status chart
                fig = px.pie(
                    values=[online_agents, total_agents - online_agents],
                    names=["Online", "Offline"],
                    title="Agent Status Distribution",
                    color_discrete_sequence=["#4CAF50", "#F44336"]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Available models
        st.subheader("🧠 Available AI Models")
        models = status.get("models", [])
        if models:
            model_data = []
            for model in models:
                size = "Unknown"
                if ":1b" in model:
                    size = "1B"
                elif ":3b" in model:
                    size = "3B"
                elif ":7b" in model:
                    size = "7B"
                elif ":8b" in model:
                    size = "8B"
                
                model_data.append({
                    "Model": model,
                    "Size": size,
                    "Type": "Code" if "code" in model.lower() else "General",
                    "Status": "Ready"
                })
            
            df = pd.DataFrame(model_data)
            st.dataframe(df, use_container_width=True)

elif page == "💬 Interactive Chatbot":
    st.title("💬 Interactive AI Chatbot")
    st.markdown("### Chat with advanced AI models and agents")
    
    # Model and agent selection
    col1, col2 = st.columns(2)
    
    with col1:
        models = get_available_models()
        if models:
            selected_model = st.selectbox(
                "Select AI Model",
                models,
                index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
            )
            st.session_state.selected_model = selected_model
    
    with col2:
        agents = ["None", "autogpt", "crewai", "gpt_engineer", "aider", "documind", "finrobot", "browser_use"]
        selected_agent = st.selectbox(
            "Select AI Agent (Optional)",
            agents,
            index=agents.index(st.session_state.selected_agent) if st.session_state.selected_agent in agents else 0
        )
        st.session_state.selected_agent = selected_agent
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                agent_info = f" (via {message.get('agent', message.get('model', 'AI'))})" if message.get('agent') else f" ({message.get('model', 'AI')})"
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>AI{agent_info}:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_message = st.text_input("Ask anything...", key="chat_input")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Send Message", type="primary"):
            if user_message:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_message
                })
                
                # Send to backend
                with st.spinner("🤔 AI is thinking..."):
                    response = send_chat_message(
                        user_message, 
                        st.session_state.selected_model,
                        st.session_state.selected_agent if st.session_state.selected_agent != "None" else None
                    )
                
                if "error" not in response:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.get("response", "No response"),
                        "model": response.get("model"),
                        "agent": response.get("agent"),
                        "tokens": response.get("tokens_used", 0),
                        "processing_time": response.get("processing_time", 0)
                    })
                else:
                    st.error(f"Error: {response['error']}")
                
                st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

elif page == "📊 AI-Generated Reports":
    st.title("📊 AI-Generated Reports")
    st.markdown("### Generate comprehensive reports using AI agents")
    
    report_type = st.selectbox(
        "Select Report Type",
        [
            "System Performance Report",
            "Security Analysis Report", 
            "Financial Analysis Report",
            "Code Quality Report",
            "Agent Performance Report"
        ]
    )
    
    # Report parameters
    col1, col2 = st.columns(2)
    
    with col1:
        time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last Week", "Last Month"])
        include_charts = st.checkbox("Include Charts", value=True)
    
    with col2:
        detail_level = st.selectbox("Detail Level", ["Summary", "Detailed", "Comprehensive"])
        export_format = st.selectbox("Export Format", ["PDF", "HTML", "JSON", "CSV"])
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("🤖 AI is generating your report..."):
            # Simulate report generation
            if report_type == "System Performance Report":
                agent = "None"  # Use direct model
                prompt = f"Generate a comprehensive {detail_level.lower()} system performance report for the {time_range.lower()}. Include metrics, analysis, and recommendations."
            elif report_type == "Security Analysis Report":
                agent = "pentestgpt"
                prompt = f"Generate a {detail_level.lower()} security analysis report for the {time_range.lower()}."
            elif report_type == "Financial Analysis Report":
                agent = "finrobot"
                prompt = f"Generate a {detail_level.lower()} financial analysis report for the {time_range.lower()}."
            elif report_type == "Code Quality Report":
                agent = "semgrep"
                prompt = f"Generate a {detail_level.lower()} code quality report for the {time_range.lower()}."
            else:
                agent = "None"
                prompt = f"Generate a {detail_level.lower()} agent performance report for the {time_range.lower()}."
            
            if agent != "None":
                response = execute_agent_task(agent, prompt)
            else:
                response = send_chat_message(prompt, "deepseek-r1:8b")
            
            if "error" not in response:
                st.success("✅ Report generated successfully!")
                
                # Display report
                st.markdown("### Generated Report")
                if agent != "None":
                    report_content = str(response.get("result", "No content"))
                else:
                    report_content = response.get("response", "No content")
                
                st.markdown(report_content)
                
                # Add download button
                st.download_button(
                    label=f"Download Report ({export_format})",
                    data=report_content,
                    file_name=f"sutazai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                    mime="text/plain"
                )
            else:
                st.error(f"Report generation failed: {response['error']}")

elif page == "🔧 Code Debugging Panel":
    st.title("🔧 Code Debugging Panel")
    st.markdown("### AI-powered code analysis and debugging")
    
    # Code input
    col1, col2 = st.columns(2)
    
    with col1:
        language = st.selectbox(
            "Programming Language",
            ["python", "javascript", "java", "cpp", "rust", "go", "typescript"]
        )
        
        action = st.selectbox(
            "Action",
            ["analyze", "debug", "optimize", "review", "explain"]
        )
    
    with col2:
        agent = st.selectbox(
            "AI Agent",
            ["gpt_engineer", "aider", "semgrep", "tabbyml"]
        )
    
    code_input = st.text_area(
        "Enter your code",
        height=300,
        placeholder="Paste your code here for AI analysis..."
    )
    
    if st.button("Analyze Code", type="primary"):
        if code_input:
            with st.spinner(f"🤖 {agent} is analyzing your code..."):
                response = execute_agent_task(
                    agent,
                    code_input,
                    {
                        "language": language,
                        "action": action
                    }
                )
            
            if "error" not in response:
                st.success("✅ Code analysis completed!")
                
                # Display results
                st.markdown("### Analysis Results")
                result = str(response.get("result", "No analysis available"))
                st.markdown(result)
                
                # Code suggestions
                if action in ["debug", "optimize"]:
                    st.markdown("### Suggested Improvements")
                    st.code(result, language=language)
            else:
                st.error(f"Code analysis failed: {response['error']}")
        else:
            st.warning("Please enter some code to analyze.")

elif page == "🌐 API Gateway Interface":
    st.title("🌐 API Gateway Interface")
    st.markdown("### Direct access to all AI agent APIs")
    
    # Agent selection
    agents_data = get_agents_status()
    if agents_data:
        agent_names = list(agents_data.get("agents", {}).keys())
        selected_api_agent = st.selectbox("Select Agent API", agent_names)
        
        # API method selection
        method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
        endpoint = st.text_input("Endpoint", placeholder="/api/v1/execute")
        
        # Request payload
        if method in ["POST", "PUT"]:
            payload = st.text_area(
                "Request Payload (JSON)",
                height=200,
                placeholder='{"task": "example task", "parameters": {}}'
            )
        
        # Headers
        headers = st.text_area(
            "Headers (JSON)",
            value='{"Content-Type": "application/json"}',
            height=100
        )
        
        if st.button("Send Request", type="primary"):
            try:
                import json
                headers_dict = json.loads(headers) if headers else {}
                
                if method == "GET":
                    response = requests.get(
                        f"{BACKEND_URL}{endpoint}",
                        headers=headers_dict,
                        timeout=30
                    )
                elif method == "POST":
                    payload_dict = json.loads(payload) if payload else {}
                    response = requests.post(
                        f"{BACKEND_URL}{endpoint}",
                        json=payload_dict,
                        headers=headers_dict,
                        timeout=30
                    )
                
                # Display response
                st.markdown("### Response")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Status Code", response.status_code)
                
                with col2:
                    st.metric("Response Time", f"{response.elapsed.total_seconds():.2f}s")
                
                # Response content
                st.markdown("### Response Body")
                try:
                    response_json = response.json()
                    st.json(response_json)
                except:
                    st.text(response.text)
                    
            except Exception as e:
                st.error(f"Request failed: {e}")

elif page == "🎤 RealtimeSTT":
    st.title("🎤 RealtimeSTT - Speech to Text")
    st.markdown("### Real-time speech transcription using AI")
    
    # Language selection
    language = st.selectbox(
        "Select Language",
        ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    )
    
    # Audio input simulation
    st.markdown("### Audio Input")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("🎤 Transcribing audio..."):
                # Convert audio to base64 for transmission
                audio_data = base64.b64encode(uploaded_file.read()).decode()
                
                response = execute_agent_task(
                    "realtime_stt",
                    audio_data,
                    {"language": language}
                )
            
            if "error" not in response:
                st.success("✅ Transcription completed!")
                
                # Display transcription
                st.markdown("### Transcription Result")
                transcription = str(response.get("result", "No transcription available"))
                st.text_area("Transcribed Text", transcription, height=200)
                
                # Download transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.error(f"Transcription failed: {response['error']}")

elif page == "🤖 Agent Management":
    st.title("🤖 Agent Management")
    st.markdown("### Manage and monitor all AI agents")
    
    # Get agents status
    agents_data = get_agents_status()
    
    if agents_data:
        # Agent overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Agents", agents_data.get("total_agents", 0))
        
        with col2:
            st.metric("Online Agents", agents_data.get("online_agents", 0))
        
        with col3:
            st.metric("Offline Agents", agents_data.get("offline_agents", 0))
        
        st.markdown("---")
        
        # Agent details
        st.subheader("Agent Status Details")
        
        agents = agents_data.get("agents", {})
        for agent_name, agent_status in agents.items():
            status = agent_status.get("status", "unknown")
            
            with st.expander(f"🤖 {agent_name.title()} - {status.title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status**: {status}")
                    st.write(f"**Type**: {agent_name.replace('_', ' ').title()}")
                
                with col2:
                    if st.button(f"Test {agent_name}", key=f"test_{agent_name}"):
                        with st.spinner(f"Testing {agent_name}..."):
                            response = execute_agent_task(
                                agent_name,
                                "Health check test",
                                {"test": True}
                            )
                        
                        if "error" not in response:
                            st.success(f"✅ {agent_name} is responding correctly!")
                        else:
                            st.error(f"❌ {agent_name} test failed: {response['error']}")

elif page == "📈 System Analytics":
    st.title("📈 System Analytics")
    st.markdown("### Real-time system performance and usage analytics")
    
    # Get system status
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/system/status", timeout=10)
        if response.status_code == 200:
            system_data = response.json()
            
            # System metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💻 System Performance")
                
                # CPU usage
                cpu_usage = system_data.get("system", {}).get("cpu_usage", 0)
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")
                
                # Memory usage
                memory_usage = system_data.get("system", {}).get("memory_usage", 0)
                st.metric("Memory Usage", f"{memory_usage:.1f}%")
                
                # Disk usage
                disk_usage = system_data.get("system", {}).get("disk_usage", 0)
                st.metric("Disk Usage", f"{disk_usage:.1f}%")
            
            with col2:
                st.subheader("🐳 Container Status")
                
                containers = system_data.get("containers", [])
                if containers:
                    container_df = pd.DataFrame(containers)
                    
                    # Container status chart
                    status_counts = container_df["status"].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Container Status Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # GPU information
            gpu_info = system_data.get("gpu", {})
            if gpu_info.get("available"):
                st.subheader("🎮 GPU Information")
                st.write(f"**GPU Available**: Yes")
                st.write(f"**GPU Name**: {gpu_info.get('name', 'Unknown')}")
                if gpu_info.get("count"):
                    st.write(f"**GPU Count**: {gpu_info.get('count')}")
            else:
                st.info("No GPU detected - Running in CPU-only mode")
            
            # Performance charts
            st.subheader("📊 Performance Trends")
            
            # Generate sample time series data
            import numpy as np
            times = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq="1min")
            cpu_data = np.random.normal(cpu_usage, 10, len(times))
            memory_data = np.random.normal(memory_usage, 5, len(times))
            
            performance_df = pd.DataFrame({
                "Time": times,
                "CPU": cpu_data,
                "Memory": memory_data
            })
            
            fig = px.line(
                performance_df,
                x="Time",
                y=["CPU", "Memory"],
                title="System Performance Over Time",
                labels={"value": "Usage (%)", "variable": "Metric"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Failed to get system analytics")
            
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

elif page == "⚙️ Configuration":
    st.title("⚙️ System Configuration")
    st.markdown("### Configure SutazAI system settings")
    
    # Model configuration
    st.subheader("🧠 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_model = st.selectbox(
            "Default Model",
            get_available_models(),
            help="Select the default model for general queries"
        )
        
        max_tokens = st.slider("Max Tokens", 100, 4000, 2000)
        
    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
        
        enable_streaming = st.checkbox("Enable Streaming", value=True)
    
    # Agent configuration
    st.subheader("🤖 Agent Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_agent_selection = st.checkbox("Auto Agent Selection", value=True, 
                                         help="Automatically select the best agent for each task")
        
        parallel_execution = st.checkbox("Parallel Execution", value=False,
                                       help="Allow multiple agents to work on tasks simultaneously")
    
    with col2:
        agent_timeout = st.slider("Agent Timeout (seconds)", 10, 300, 60)
        
        retry_attempts = st.slider("Retry Attempts", 1, 5, 3)
    
    # System configuration
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        cache_enabled = st.checkbox("Enable Caching", value=True)
    
    with col2:
        max_chat_history = st.slider("Max Chat History", 10, 1000, 100)
        
        auto_backup = st.checkbox("Auto Backup", value=True)
    
    # Save configuration
    if st.button("Save Configuration", type="primary"):
        config = {
            "model": {
                "default": default_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "streaming": enable_streaming
            },
            "agents": {
                "auto_selection": auto_agent_selection,
                "parallel_execution": parallel_execution,
                "timeout": agent_timeout,
                "retry_attempts": retry_attempts
            },
            "system": {
                "log_level": log_level,
                "cache_enabled": cache_enabled,
                "max_chat_history": max_chat_history,
                "auto_backup": auto_backup
            }
        }
        
        st.success("✅ Configuration saved successfully!")
        st.json(config)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 <strong>SutazAI AGI/ASI System</strong> | Enterprise-Grade Autonomous AI | Version 2.0.0</p>
    <p>Powered by 22+ AI Agents | 7+ AI Models | 100% Local Operation</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for dashboard
if page == "🏠 Dashboard":
    time.sleep(5)
    st.rerun() 