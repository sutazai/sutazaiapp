#!/usr/bin/env python3
"""
SutazAI Frontend Application
============================

Modern, responsive Streamlit interface for the SutazAI AGI/ASI system.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configure page
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = "http://backend:8000"
REFRESH_INTERVAL = 5  # seconds

# Utility functions
@st.cache_data(ttl=REFRESH_INTERVAL)
def get_system_status() -> Dict[str, Any]:
    """Get system status from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/system/status", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Backend connection failed: {e}")
        return {"status": "error", "error": str(e)}

@st.cache_data(ttl=30)
def get_available_models() -> List[Dict[str, Any]]:
    """Get available AI models"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/models", timeout=10)
        response.raise_for_status()
        return response.json().get("models", [])
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return []

def send_chat_message(message: str, model: str) -> Dict[str, Any]:
    """Send chat message to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": message,
                "model": model,
                "temperature": 0.7,
                "max_tokens": 2048
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def create_task(description: str, task_type: str) -> Dict[str, Any]:
    """Create a new task"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/tasks",
            json={
                "description": description,
                "task_type": task_type,
                "priority": 5
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Sidebar
def render_sidebar():
    """Render sidebar with system information"""
    st.sidebar.title("🤖 SutazAI Control Panel")
    
    # System status
    st.sidebar.subheader("System Status")
    status_data = get_system_status()
    
    if status_data.get("status") == "error":
        st.sidebar.error("System Offline")
        st.sidebar.write(f"Error: {status_data.get('error', 'Unknown error')}")
    else:
        st.sidebar.success("System Online")
        
        # System metrics
        if "system_metrics" in status_data:
            metrics = status_data["system_metrics"]
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                st.metric("CPU", f"{metrics.get('cpu_usage', 0):.1f}%")
                st.metric("Memory", f"{metrics.get('memory_usage', 0):.1f}%")
            
            with col2:
                st.metric("Disk", f"{metrics.get('disk_usage', 0):.1f}%")
                uptime = status_data.get('uptime_seconds', 0)
                st.metric("Uptime", f"{uptime/3600:.1f}h")
        
        # Agent status
        agents = status_data.get("agents", [])
        if agents:
            idle_agents = len([a for a in agents if isinstance(a, dict) and a.get('status') == 'idle'])
            total_agents = len(agents)
            st.sidebar.metric("Active Agents", f"{idle_agents}/{total_agents}")
    
    # Navigation
    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "Navigation",
        ["Chat Interface", "Agent Management", "System Monitoring", "Task Management"]
    )

# Main pages
def render_chat_interface():
    """Render chat interface"""
    st.title("💬 Intelligent Chat Interface")
    st.markdown("Chat with SutazAI's advanced AI models")
    
    # Model selection
    models = get_available_models()
    if models:
        model_names = [m.get("name", "unknown") for m in models]
        selected_model = st.selectbox("Select Model", model_names, index=0)
    else:
        selected_model = "llama3"
        st.warning("Could not load models from backend. Using default model.")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "metadata" in message:
                with st.expander("Details"):
                    st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(prompt, selected_model)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    st.write(response.get("response", "No response"))
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("response", "No response"),
                        "metadata": {
                            "model": response.get("model"),
                            "tokens_used": response.get("tokens_used"),
                            "processing_time": response.get("processing_time")
                        }
                    })

def render_agent_management():
    """Render agent management interface"""
    st.title("🤖 Agent Management")
    st.markdown("Monitor and manage AI agents")
    
    status_data = get_system_status()
    agents = status_data.get("agents", [])
    
    if not agents:
        st.info("No agents currently active")
        return
    
    # Agent statistics
    if isinstance(agents[0], dict):
        agent_stats = {}
        for agent in agents:
            agent_type = agent.get("type", "unknown")
            status = agent.get("status", "unknown")
            
            if agent_type not in agent_stats:
                agent_stats[agent_type] = {"total": 0, "idle": 0, "busy": 0, "error": 0}
            
            agent_stats[agent_type]["total"] += 1
            agent_stats[agent_type][status] += 1
        
        # Display statistics
        st.subheader("Agent Statistics")
        
        cols = st.columns(len(agent_stats))
        for i, (agent_type, stats) in enumerate(agent_stats.items()):
            with cols[i]:
                st.metric(
                    f"{agent_type.title()}",
                    f"{stats['idle']}/{stats['total']}",
                    f"{stats['busy']} busy"
                )
        
        # Agent details table
        st.subheader("Agent Details")
        agent_df = pd.DataFrame([
            {
                "ID": agent.get("id", "")[:12],
                "Name": agent.get("name", ""),
                "Type": agent.get("type", ""),
                "Status": agent.get("status", ""),
                "Current Task": agent.get("current_task", "None"),
                "Completed Tasks": agent.get("completed_tasks", 0)
            }
            for agent in agents
        ])
        
        st.dataframe(agent_df, use_container_width=True)

def render_system_monitoring():
    """Render system monitoring interface"""
    st.title("📊 System Monitoring")
    st.markdown("Real-time system performance and health monitoring")
    
    status_data = get_system_status()
    
    if "system_metrics" not in status_data:
        st.warning("System metrics not available")
        return
    
    metrics = status_data["system_metrics"]
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = metrics.get("cpu_usage", 0)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%", 
                 delta=f"{'High' if cpu_usage > 80 else 'Normal'}")
    
    with col2:
        memory_usage = metrics.get("memory_usage", 0)
        st.metric("Memory Usage", f"{memory_usage:.1f}%",
                 delta=f"{'High' if memory_usage > 85 else 'Normal'}")
    
    with col3:
        disk_usage = metrics.get("disk_usage", 0)
        st.metric("Disk Usage", f"{disk_usage:.1f}%",
                 delta=f"{'High' if disk_usage > 90 else 'Normal'}")
    
    with col4:
        load_avg = metrics.get("load_average", [0, 0, 0])
        st.metric("Load Average", f"{load_avg[0]:.2f}",
                 delta=f"{load_avg[1]:.2f} (5m)")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Resource usage pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Used', 'Free'],
            values=[memory_usage, 100 - memory_usage],
            title="Memory Usage"
        )])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # System load gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cpu_usage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU Usage (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

def render_task_management():
    """Render task management interface"""
    st.title("📋 Task Management")
    st.markdown("Create and monitor system tasks")
    
    # Task creation
    st.subheader("Create New Task")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        task_description = st.text_area(
            "Task Description",
            placeholder="Describe what you want the AI to do..."
        )
    
    with col2:
        task_type = st.selectbox(
            "Task Type",
            ["chat", "code_generation", "document_processing", "web_automation", "data_analysis"]
        )
        
        if st.button("Create Task", type="primary"):
            if task_description:
                result = create_task(task_description, task_type)
                
                if "error" in result:
                    st.error(f"Error creating task: {result['error']}")
                else:
                    st.success(f"Task created with ID: {result.get('task_id', 'unknown')}")
            else:
                st.warning("Please enter a task description")
    
    # Task status
    st.subheader("Recent Tasks")
    status_data = get_system_status()
    tasks = status_data.get("tasks", [])
    
    if tasks:
        task_df = pd.DataFrame([
            {
                "ID": task.get("id", "")[:12],
                "Type": task.get("type", ""),
                "Status": task.get("status", ""),
                "Created": task.get("created_at", ""),
                "Agent": task.get("agent_id", "")[:12] if task.get("agent_id") else "None"
            }
            for task in tasks[:20]  # Show last 20 tasks
        ])
        
        st.dataframe(task_df, use_container_width=True)
    else:
        st.info("No recent tasks")

# Main application
def main():
    """Main application entry point"""
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.98);
    }
    h1 {
        color: #2E3440;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Auto-refresh
    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # Render selected page
    if selected_page == "Chat Interface":
        render_chat_interface()
    elif selected_page == "Agent Management":
        render_agent_management()
    elif selected_page == "System Monitoring":
        render_system_monitoring()
    elif selected_page == "Task Management":
        render_task_management()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**SutazAI AGI/ASI System** | "
        f"Backend: {BACKEND_URL} | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()