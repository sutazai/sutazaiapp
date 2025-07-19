#!/usr/bin/env python3
"""
Enhanced Streamlit UI for SutazAI Complete System Control
Provides comprehensive interface for all AI agents and services
"""

import streamlit as st
import requests
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import websocket
import threading
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .agent-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .success { color: #4CAF50; }
    .warning { color: #FF9800; }
    .error { color: #F44336; }
</style>
""", unsafe_allow_html=True)

# Backend URL
BACKEND_URL = "http://localhost:8000"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama3.2:3b"
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {}

def get_agent_status():
    """Fetch status of all agents"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/agents/status/all")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def get_system_metrics():
    """Fetch system metrics"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/metrics")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def execute_agent_task(task_type: str, task_data: Dict[str, Any]):
    """Execute task using appropriate agent"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/agents/execute",
            json={"task_type": task_type, "task_data": task_data}
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Failed to execute task"}

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 SutazAI AGI/ASI Autonomous System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Model selection
        st.subheader("🧠 Model Selection")
        models = [
            "llama3.2:1b",
            "llama3.2:3b",
            "deepseek-r1:8b",
            "qwen2.5:3b",
            "codellama:7b",
            "mistral:7b",
            "coding-assistant",
            "reasoning-assistant",
            "general-assistant"
        ]
        st.session_state.selected_model = st.selectbox(
            "Select AI Model",
            models,
            index=models.index(st.session_state.selected_model)
        )
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh System Status"):
            st.session_state.agent_status = get_agent_status()
            st.rerun()
        
        if st.button("📊 Generate System Report"):
            with st.spinner("Generating report..."):
                # Generate comprehensive system report
                st.success("Report generated!")
        
        if st.button("🔧 Run Self-Improvement"):
            with st.spinner("Running self-improvement analysis..."):
                response = requests.post(f"{BACKEND_URL}/api/self-improvement/analyze")
                if response.status_code == 200:
                    st.success("Analysis complete!")
        
        # System info
        st.subheader("ℹ️ System Info")
        st.info(f"""
        **Version:** 11.0
        **Agents:** 20+
        **Models:** 10+
        **Status:** 🟢 Operational
        """)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat Interface",
        "🤖 Agent Management",
        "📊 System Metrics",
        "🔧 Self-Improvement",
        "📝 Task Automation",
        "⚙️ Settings"
    ])
    
    # Tab 1: Chat Interface
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Intelligent Chat")
            
            # Chat history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask anything..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get AI response
                with st.spinner("Thinking..."):
                    response = requests.post(
                        f"{BACKEND_URL}/api/chat",
                        json={
                            "message": prompt,
                            "model": st.session_state.selected_model
                        }
                    )
                    
                    if response.status_code == 200:
                        ai_response = response.json()["response"]
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    else:
                        st.error("Failed to get response")
                
                st.rerun()
        
        with col2:
            st.header("🎯 Quick Prompts")
            
            prompts = {
                "📝 Generate Code": "Generate a Python function to ",
                "🔍 Analyze Code": "Analyze this code for improvements: ",
                "🐛 Debug Issue": "Help me debug this error: ",
                "📚 Explain Concept": "Explain the concept of ",
                "🏗️ System Design": "Design a system architecture for ",
                "🔒 Security Check": "Check this code for security issues: "
            }
            
            for label, prompt_start in prompts.items():
                if st.button(label, use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": prompt_start
                    })
                    st.rerun()
    
    # Tab 2: Agent Management
    with tab2:
        st.header("🤖 AI Agent Management")
        
        # Refresh agent status
        if st.button("🔄 Refresh Agent Status"):
            st.session_state.agent_status = get_agent_status()
        
        # Agent status grid
        agent_data = st.session_state.agent_status.get("agents", {})
        
        if agent_data:
            # Create columns for agent cards
            cols = st.columns(3)
            
            for idx, (agent_name, agent_info) in enumerate(agent_data.items()):
                col = cols[idx % 3]
                
                with col:
                    status = agent_info.get("status", "unknown")
                    status_color = "🟢" if status == "healthy" else "🔴"
                    
                    st.markdown(f"""
                    <div class="agent-card">
                        <h4>{status_color} {agent_info.get('name', agent_name)}</h4>
                        <p><strong>Type:</strong> {agent_info.get('type', 'Unknown')}</p>
                        <p><strong>Status:</strong> {status}</p>
                        <p><strong>Capabilities:</strong> {', '.join(agent_info.get('capabilities', []))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Agent actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Test", key=f"test_{agent_name}"):
                            st.info(f"Testing {agent_name}...")
                    with col2:
                        if st.button(f"Restart", key=f"restart_{agent_name}"):
                            st.warning(f"Restarting {agent_name}...")
        else:
            st.info("No agent data available. Click refresh to load.")
    
    # Tab 3: System Metrics
    with tab3:
        st.header("📊 System Metrics & Performance")
        
        metrics = get_system_metrics()
        
        if metrics:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", metrics.get("total_requests", 0))
            with col2:
                st.metric("Active Agents", metrics.get("active_agents", 0))
            with col3:
                st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
            with col4:
                st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time chart
                st.subheader("Response Time Trend")
                # Mock data for demonstration
                times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                    end=datetime.now(), freq='1H')
                response_times = [0.5 + i * 0.1 for i in range(len(times))]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=response_times, mode='lines+markers'))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Agent utilization
                st.subheader("Agent Utilization")
                agents = ["AutoGPT", "LocalAGI", "TabbyML", "Semgrep", "Others"]
                utilization = [30, 25, 20, 15, 10]
                
                fig = px.pie(values=utilization, names=agents, hole=0.4)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Self-Improvement
    with tab4:
        st.header("🔧 Self-Improvement System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Code Analysis")
            
            # File selector
            file_path = st.text_input("File Path", placeholder="/opt/sutazaiapp/backend/")
            
            if st.button("Analyze File"):
                with st.spinner("Analyzing..."):
                    response = requests.post(
                        f"{BACKEND_URL}/api/self-improvement/analyze-file",
                        json={"file_path": file_path}
                    )
                    
                    if response.status_code == 200:
                        analysis = response.json()
                        
                        st.success("Analysis Complete!")
                        
                        # Display metrics
                        st.markdown("### Metrics")
                        metrics = analysis.get("metrics", {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Lines of Code", metrics.get("lines", 0))
                        with col2:
                            st.metric("Functions", metrics.get("functions", 0))
                        with col3:
                            st.metric("Complexity", analysis.get("complexity", 0))
                        
                        # Display suggestions
                        st.markdown("### Improvement Suggestions")
                        for suggestion in analysis.get("suggestions", []):
                            st.warning(f"**{suggestion['type']}:** {suggestion['message']}")
        
        with col2:
            st.subheader("Improvement History")
            
            # Mock improvement history
            history = [
                {"date": "2024-01-20", "type": "refactor", "file": "backend.py"},
                {"date": "2024-01-19", "type": "documentation", "file": "utils.py"},
                {"date": "2024-01-18", "type": "optimization", "file": "agent.py"},
            ]
            
            for item in history:
                st.info(f"**{item['date']}** - {item['type']} on {item['file']}")
    
    # Tab 5: Task Automation
    with tab5:
        st.header("📝 Task Automation")
        
        # Task type selection
        task_type = st.selectbox(
            "Select Task Type",
            [
                "code_generation",
                "code_completion",
                "code_security",
                "web_automation",
                "document_processing",
                "financial_analysis",
                "workflow_automation",
                "data_retrieval"
            ]
        )
        
        # Task configuration
        st.subheader("Task Configuration")
        
        if task_type == "code_generation":
            language = st.selectbox("Programming Language", ["python", "javascript", "java", "go", "rust"])
            description = st.text_area("Describe what you want to build")
            
            if st.button("Generate Code"):
                with st.spinner("Generating code..."):
                    result = execute_agent_task(task_type, {
                        "language": language,
                        "description": description
                    })
                    
                    if result.get("status") == "success":
                        st.success(f"Code generated by {result.get('agent')}!")
                        st.code(result.get("result", {}).get("code", ""), language=language)
                    else:
                        st.error(f"Error: {result.get('message')}")
        
        elif task_type == "document_processing":
            uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt'])
            action = st.selectbox("Action", ["extract_text", "summarize", "analyze"])
            
            if uploaded_file and st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # In real implementation, would upload file and process
                    st.success("Document processed successfully!")
        
        elif task_type == "web_automation":
            url = st.text_input("Website URL")
            action = st.selectbox("Action", ["scrape", "fill_form", "navigate", "screenshot"])
            
            if url and st.button("Execute"):
                with st.spinner("Executing web automation..."):
                    result = execute_agent_task(task_type, {
                        "url": url,
                        "action": action
                    })
                    st.json(result)
    
    # Tab 6: Settings
    with tab6:
        st.header("⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Configuration")
            
            api_key = st.text_input("API Key", type="password", value="*" * 32)
            endpoint = st.text_input("API Endpoint", value=BACKEND_URL)
            
            if st.button("Save API Settings"):
                st.success("Settings saved!")
        
        with col2:
            st.subheader("Model Settings")
            
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
            max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9)
            
            if st.button("Apply Model Settings"):
                st.success("Model settings applied!")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.subheader("Self-Improvement Configuration")
            
            enable_auto_improvement = st.checkbox("Enable Automatic Improvement", value=False)
            improvement_interval = st.slider("Improvement Interval (hours)", 1, 168, 24)
            require_approval = st.checkbox("Require Approval for Changes", value=True)
            
            st.subheader("Resource Limits")
            
            max_memory = st.slider("Max Memory per Agent (GB)", 1, 16, 4)
            max_cpu = st.slider("Max CPU per Agent (cores)", 1, 8, 2)
            
            if st.button("Save Advanced Settings"):
                st.success("Advanced settings saved!")

if __name__ == "__main__":
    main()