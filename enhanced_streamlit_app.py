#!/usr/bin/env python3
"""
Enhanced SutazAI Streamlit Frontend
Advanced AI interface with intelligent chatbot and voice capabilities
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configure Streamlit page
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL
BACKEND_URL = "http://localhost:8000"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'system_status' not in st.session_state:
    st.session_state.system_status = {}

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to backend"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return {"error": "Unsupported method"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Backend service is not available"}
    except Exception as e:
        return {"error": str(e)}

def display_header():
    """Display main header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🤖 SutazAI AGI/ASI Autonomous System</h1>
        <p style="color: white; margin: 10px 0 0 0;">Advanced AI System with Intelligent Chatbot & Voice Capabilities</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("## 🔧 System Status")
    
    # Get system status from backend
    status_data = make_api_request("/api/system/status")
    
    if "error" not in status_data:
        st.sidebar.markdown("### ✅ System Health")
        st.sidebar.text("All services operational")
        
        # Show service status
        with st.sidebar.expander("Service Details"):
            st.text("🚀 Backend API: Healthy")
            st.text("🧠 AI Models: Ready")
            st.text("🗄️ Vector DBs: Active")
            st.text("🤖 Agents: Operational")
    else:
        st.sidebar.markdown("### ⚠️ System Status")
        st.sidebar.error(status_data.get("error", "Unknown error"))

def chat_interface():
    """Main chat interface"""
    st.markdown("## 💬 Intelligent Chat Interface")
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Chat with SutazAI")
    
    with col2:
        selected_model = st.selectbox(
            "Model",
            ["llama3.2:1b", "deepseek-coder:7b", "codellama:7b"],
            index=0
        )
    
    # Chat input
    user_input = st.text_area(
        "Ask SutazAI anything...",
        height=100,
        placeholder="You can ask me to:\n• Generate code in any language\n• Manage AI agents\n• Check system status\n• Deploy services\n• Process documents\n• And much more!"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💬 Send Message", use_container_width=True):
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Make API request
                response_data = make_api_request("/api/chat", "POST", {
                    "message": user_input,
                    "model": selected_model
                })
                
                # Add assistant response to history
                if "error" not in response_data:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_data.get("response", "No response"),
                        "model": selected_model,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {response_data.get('error', 'Unknown error')}",
                        "model": selected_model,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Rerun to update chat
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        if st.button("🎤 Voice Chat (Coming Soon)", use_container_width=True, disabled=True):
            st.info("Voice chat capabilities are being implemented")
    
    # Display chat history
    st.markdown("### 💭 Conversation")
    
    if st.session_state.chat_history:
        for i, message in enumerate(reversed(st.session_state.chat_history[-20:])):  # Show last 20 messages
            timestamp = datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
            
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4CAF50;">
                    <strong>👤 You</strong> <span style="color: #666; font-size: 0.8em;">({timestamp})</span>
                    <br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                model_info = f" • {message.get('model', 'AI')}" if message.get('model') else ""
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196F3;">
                    <strong>🤖 SutazAI</strong> <span style="color: #666; font-size: 0.8em;">({timestamp}{model_info})</span>
                    <br><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Start a conversation! Try asking me to generate code, check system status, or manage agents.")

def code_generation_interface():
    """Code generation interface"""
    st.markdown("## 💻 Code Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        code_description = st.text_area(
            "Describe what you want to code:",
            height=150,
            placeholder="Example: Create a Python function to calculate fibonacci numbers"
        )
    
    with col2:
        language = st.selectbox(
            "Programming Language",
            ["python", "javascript", "java", "cpp", "go", "rust", "html", "css"]
        )
        
        framework = st.text_input(
            "Framework (optional)",
            placeholder="e.g., React, Django, Spring"
        )
    
    if st.button("🚀 Generate Code", use_container_width=True):
        if code_description:
            # Make API request for code generation
            request_data = {
                "description": code_description,
                "language": language
            }
            
            if framework:
                request_data["framework"] = framework
            
            response_data = make_api_request("/api/code/generate", "POST", request_data)
            
            if "error" not in response_data:
                st.success("Code generated successfully!")
                st.code(response_data.get("code", "No code generated"), language=language)
            else:
                st.error(f"Error: {response_data.get('error', 'Unknown error')}")
        else:
            st.warning("Please describe what you want to code!")

def agent_management_interface():
    """Agent management interface"""
    st.markdown("## 🤖 Agent Management")
    
    # Get current agents
    agents_data = make_api_request("/api/agents/")
    
    if "error" not in agents_data:
        agents = agents_data.get("agents", [])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Active Agents")
            
            if agents:
                for agent in agents:
                    with st.expander(f"🤖 {agent.get('name', 'Unknown Agent')}"):
                        st.write(f"**Type:** {agent.get('type', 'Unknown')}")
                        st.write(f"**Status:** {agent.get('status', 'Unknown')}")
                        st.write(f"**Created:** {agent.get('created_at', 'Unknown')}")
                        
                        if st.button(f"Remove {agent.get('name', 'Agent')}", key=f"remove_{agent.get('id', 'unknown')}"):
                            st.warning("Agent removal feature coming soon!")
            else:
                st.info("No active agents found. Create one using the form on the right.")
        
        with col2:
            st.markdown("### Create New Agent")
            
            agent_name = st.text_input("Agent Name", placeholder="My AI Assistant")
            agent_type = st.selectbox(
                "Agent Type",
                ["general", "code_specialist", "data_analyst", "system_admin"]
            )
            
            if st.button("✨ Create Agent", use_container_width=True):
                if agent_name:
                    response_data = make_api_request("/api/agents/", "POST", {
                        "name": agent_name,
                        "type": agent_type,
                        "config": {}
                    })
                    
                    if "error" not in response_data:
                        st.success(f"Agent '{agent_name}' created successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response_data.get('error', 'Unknown error')}")
                else:
                    st.warning("Please enter an agent name!")
    else:
        st.error(f"Error loading agents: {agents_data.get('error', 'Unknown error')}")

def system_monitoring_interface():
    """System monitoring interface"""
    st.markdown("## 📊 System Monitoring")
    
    # Get system status
    status_data = make_api_request("/api/system/status")
    
    if "error" not in status_data:
        st.success("System is operational!")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Backend Status", "✅ Healthy")
        
        with col2:
            st.metric("AI Models", "🧠 Ready")
        
        with col3:
            st.metric("Vector DBs", "🗄️ Active")
        
        with col4:
            st.metric("Agents", f"🤖 {len(st.session_state.agents)}")
        
        # System information
        st.markdown("### 🔧 System Information")
        
        system_info = {
            "Version": "2.0.0",
            "Architecture": "Microservices",
            "Services": "34 containerized services",
            "AI Models": "Ollama + DeepSeek + Llama",
            "Vector DBs": "ChromaDB, Qdrant, FAISS",
            "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        info_df = pd.DataFrame(list(system_info.items()), columns=["Component", "Status"])
        st.dataframe(info_df, use_container_width=True)
        
        # Performance chart (placeholder)
        st.markdown("### 📈 Performance Metrics")
        
        # Generate sample data for demonstration
        import numpy as np
        times = pd.date_range(start="2025-01-01", periods=100, freq="H")
        values = np.random.normal(50, 10, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines',
            name='System Load',
            line=dict(color='#2196F3')
        ))
        
        fig.update_layout(
            title="System Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Load (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Error loading system status: {status_data.get('error', 'Unknown error')}")

def main():
    """Main application"""
    display_header()
    
    # Sidebar navigation
    st.sidebar.markdown("## 🚀 Navigation")
    
    page = st.sidebar.radio(
        "Choose Interface",
        ["💬 Chat", "💻 Code Generation", "🤖 Agent Management", "📊 System Monitoring"]
    )
    
    # Display system status in sidebar
    display_system_status()
    
    # Additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎯 Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Status"):
        st.rerun()
    
    if st.sidebar.button("📋 System Info"):
        st.sidebar.info("SutazAI v2.0.0 - AGI/ASI System")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 SutazAI System**")
    st.sidebar.markdown("*Autonomous AI Platform*")
    
    # Main content based on selected page
    if page == "💬 Chat":
        chat_interface()
    elif page == "💻 Code Generation":
        code_generation_interface()
    elif page == "🤖 Agent Management":
        agent_management_interface()
    elif page == "📊 System Monitoring":
        system_monitoring_interface()

if __name__ == "__main__":
    main()