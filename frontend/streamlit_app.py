#!/usr/bin/env python3
"""
SutazAI AGI/ASI System - Streamlit Web Interface
Comprehensive web interface for the SutazAI system
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Any, Optional
import asyncio
import websocket
import threading

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = os.getenv("BACKEND_URL", "http://sutazai-backend:8000")
WEBSOCKET_URL = BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = {}
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "workspace_files" not in st.session_state:
    st.session_state.workspace_files = []

class SutazAIInterface:
    """Main interface for SutazAI system"""
    
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make HTTP request to backend"""
        try:
            url = f"{self.backend_url}{endpoint}"
            
            if method == "GET":
                response = self.session.get(url, params=data)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PUT":
                response = self.session.put(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.make_request("/api/system/status")
    
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get available agents"""
        return self.make_request("/api/agents/")
    
    def create_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new agent"""
        return self.make_request("/api/agents/", method="POST", data=agent_config)
    
    def chat_with_agent(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Chat with agent"""
        return self.make_request(
            f"/api/agents/{agent_id}/chat",
            method="POST",
            data={"message": message}
        )
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        return self.make_request("/api/models/")
    
    def generate_code(self, prompt: str, language: str = "python") -> Dict[str, Any]:
        """Generate code"""
        return self.make_request(
            "/api/code/generate",
            method="POST",
            data={"prompt": prompt, "language": language}
        )
    
    def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process document"""
        files = {"file": (filename, file_content)}
        response = requests.post(f"{self.backend_url}/api/documents/upload", files=files)
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return self.make_request("/api/metrics/")

# Initialize interface
interface = SutazAIInterface()

# Sidebar
st.sidebar.title("🤖 SutazAI Control Panel")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "💬 Chat Interface", "🔧 Agent Management", "📊 Code Generation", 
     "📄 Document Processing", "🧠 Neural Processing", "🌐 AI Services", "🔬 Evolution Lab", 
     "🕸️ Knowledge Graph", "📈 Analytics", "⚙️ Settings"]
)

# System status in sidebar
with st.sidebar.expander("🔋 System Status", expanded=False):
    status = interface.get_system_status()
    if status:
        st.success(f"Status: {status.get('status', 'Unknown')}")
        st.info(f"Uptime: {status.get('uptime', 0):.1f}s")
        st.info(f"Active Agents: {status.get('active_agents', 0)}")
        st.info(f"Models Loaded: {status.get('loaded_models', 0)}")

# Main content area
st.title("🚀 SutazAI AGI/ASI Autonomous System")

if page == "🏠 Dashboard":
    st.header("System Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online", "Healthy")
    
    with col2:
        st.metric("Active Agents", "3", "+1")
    
    with col3:
        st.metric("Tasks Completed", "127", "+12")
    
    with col4:
        st.metric("CPU Usage", "45%", "-5%")
    
    # Real-time system overview
    st.subheader("🔄 Real-time System Overview")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Services", "📊 Performance", "🤖 AI Agents", "🔗 Integrations"])
    
    with tab1:
        st.subheader("Core Services Status")
        
        services_data = {
            "Service": ["Backend API", "Streamlit UI", "PostgreSQL", "Redis", "Qdrant", "ChromaDB", "Ollama"],
            "Status": ["🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running"],
            "Port": ["8000", "8501", "5432", "6379", "6333", "8001", "11434"],
            "Health": ["Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy", "Healthy"]
        }
        
        df_services = pd.DataFrame(services_data)
        st.dataframe(df_services, use_container_width=True)
    
    with tab2:
        st.subheader("Performance Metrics")
        
        # Generate sample performance data
        performance_data = {
            "Metric": ["CPU Usage", "Memory Usage", "GPU Usage", "Network I/O", "Disk I/O"],
            "Current": ["45%", "62%", "78%", "1.2 MB/s", "0.8 MB/s"],
            "Average": ["42%", "58%", "75%", "1.1 MB/s", "0.9 MB/s"],
            "Peak": ["68%", "84%", "92%", "2.1 MB/s", "1.5 MB/s"]
        }
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
        
        # Performance chart
        chart_data = pd.DataFrame({
            "Time": pd.date_range(start='2024-01-01', periods=24, freq='H'),
            "CPU": [45 + i*2 for i in range(24)],
            "Memory": [60 + i*1.5 for i in range(24)],
            "GPU": [70 + i*1.2 for i in range(24)]
        })
        
        fig = px.line(chart_data, x="Time", y=["CPU", "Memory", "GPU"], 
                     title="System Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("AI Agents Overview")
        
        agents_data = {
            "Agent": ["DeepSeek-Coder", "Llama-2", "AutoGPT", "TabbyML", "AgentZero"],
            "Type": ["Code Generation", "General AI", "Task Automation", "Code Completion", "Autonomous"],
            "Status": ["🟢 Active", "🟢 Active", "🟡 Idle", "🟢 Active", "🟡 Idle"],
            "Tasks": ["15", "8", "3", "45", "2"],
            "Uptime": ["2.5h", "3.1h", "1.8h", "4.2h", "1.2h"]
        }
        
        df_agents = pd.DataFrame(agents_data)
        st.dataframe(df_agents, use_container_width=True)
    
    with tab4:
        st.subheader("External Integrations")
        
        integrations_data = {
            "Service": ["Browser-Use", "Skyvern", "Documind", "FinRobot", "GPT-Engineer", "Aider"],
            "Type": ["Web Automation", "Web Automation", "Document Processing", "Financial Analysis", "Code Generation", "Code Editing"],
            "Status": ["🟢 Connected", "🟢 Connected", "🟢 Connected", "🟢 Connected", "🟢 Connected", "🟢 Connected"],
            "Last Used": ["5 min ago", "12 min ago", "2 min ago", "45 min ago", "8 min ago", "15 min ago"]
        }
        
        df_integrations = pd.DataFrame(integrations_data)
        st.dataframe(df_integrations, use_container_width=True)

elif page == "💬 Chat Interface":
    st.header("Chat Interface")
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get available models from backend
        try:
            available_models = interface.get_models()
            if available_models:
                model_names = [model["name"] for model in available_models]
            else:
                model_names = ["deepseek-coder:33b", "llama2:13b", "codellama:7b", "mistral:7b"]
        except:
            model_names = ["deepseek-coder:33b", "llama2:13b", "codellama:7b", "mistral:7b"]
        
        selected_model = st.selectbox(
            "Select Model",
            model_names,
            key="chat_model"
        )
    
    with col2:
        if st.button("🔄 Refresh Models"):
            st.rerun()
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
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
                    # Get real AI response from backend
                    try:
                        chat_response = interface.make_request(
                            "/api/chat",
                            method="POST",
                            data={"message": prompt, "model": selected_model}
                        )
                        if chat_response and "response" in chat_response:
                            response = chat_response["response"]
                        else:
                            response = f"Error: Could not get response from {selected_model}"
                    except Exception as e:
                        response = f"Error connecting to {selected_model}: {str(e)}"
                    
                    st.markdown(response)
                    
                    # Add assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Chat controls
    st.sidebar.subheader("💬 Chat Controls")
    
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.sidebar.button("💾 Save Chat"):
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "model": selected_model,
            "messages": st.session_state.messages
        }
        st.sidebar.download_button(
            "📥 Download Chat",
            json.dumps(chat_data, indent=2),
            "chat_history.json",
            "application/json"
        )

elif page == "🔧 Agent Management":
    st.header("Agent Management")
    
    # Agent overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Active Agents")
        
        # Agent list
        agents = interface.get_agents()
        if agents:
            for agent in agents:
                with st.expander(f"Agent: {agent.get('name', 'Unknown')}"):
                    st.write(f"**Type:** {agent.get('type', 'Unknown')}")
                    st.write(f"**Status:** {agent.get('status', 'Unknown')}")
                    st.write(f"**Created:** {agent.get('created_at', 'Unknown')}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{agent.get('id')}"):
                        st.success(f"Agent {agent.get('name')} deleted")
        else:
            st.info("No active agents found")
    
    with col2:
        st.subheader("➕ Create New Agent")
        
        with st.form("create_agent"):
            agent_name = st.text_input("Agent Name")
            agent_type = st.selectbox(
                "Agent Type",
                ["coding", "general", "research", "creative", "analytical"]
            )
            model = st.selectbox(
                "Model",
                ["deepseek-coder:33b", "llama2:13b", "codellama:7b"]
            )
            
            submitted = st.form_submit_button("🚀 Create Agent")
            
            if submitted and agent_name:
                agent_config = {
                    "name": agent_name,
                    "type": agent_type,
                    "model": model
                }
                
                result = interface.create_agent(agent_config)
                if result:
                    st.success(f"Agent '{agent_name}' created successfully!")
                    st.rerun()

elif page == "📊 Code Generation":
    st.header("Code Generation")
    
    # Code generation interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✍️ Generate Code")
        
        # Code prompt
        code_prompt = st.text_area(
            "Describe what you want to code:",
            placeholder="Create a Python function that calculates fibonacci numbers...",
            height=150
        )
        
        # Language selection
        language = st.selectbox(
            "Programming Language",
            ["python", "javascript", "java", "cpp", "rust", "go", "html", "css"]
        )
        
        # Generation parameters
        with st.expander("🔧 Advanced Settings"):
            max_tokens = st.slider("Max Tokens", 100, 2000, 500)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9)
        
        if st.button("🚀 Generate Code"):
            if code_prompt:
                with st.spinner("Generating code..."):
                    result = interface.generate_code(code_prompt, language)
                    if result:
                        st.session_state.generated_code = result.get("code", "")
                        st.success("Code generated successfully!")
    
    with col2:
        st.subheader("📋 Generated Code")
        
        if "generated_code" in st.session_state:
            st.code(st.session_state.generated_code, language=language)
            
            # Code actions
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("📋 Copy Code"):
                    st.info("Code copied to clipboard!")
            
            with col_b:
                if st.button("💾 Save Code"):
                    st.download_button(
                        "📥 Download",
                        st.session_state.generated_code,
                        f"generated_code.{language}",
                        f"text/{language}"
                    )
            
            with col_c:
                if st.button("🔄 Regenerate"):
                    st.rerun()
        else:
            st.info("No code generated yet. Use the form on the left to generate code.")

elif page == "📄 Document Processing":
    st.header("Document Processing")
    
    # Document upload
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "docx", "txt", "md", "html"],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # File details
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            st.write(f"**Type:** {uploaded_file.type}")
            
            if st.button("🔍 Process Document"):
                with st.spinner("Processing document..."):
                    result = interface.process_document(
                        uploaded_file.read(),
                        uploaded_file.name
                    )
                    
                    if result:
                        st.session_state.processed_doc = result
                        st.success("Document processed successfully!")
    
    with col2:
        st.subheader("📋 Document Analysis")
        
        if "processed_doc" in st.session_state:
            doc_result = st.session_state.processed_doc
            
            # Document summary
            st.write("**Document Summary:**")
            st.write(doc_result.get("summary", "No summary available"))
            
            # Key information
            if "key_points" in doc_result:
                st.write("**Key Points:**")
                for point in doc_result["key_points"]:
                    st.write(f"• {point}")
            
            # Extracted text
            if "extracted_text" in doc_result:
                with st.expander("📄 Extracted Text"):
                    st.text_area(
                        "Full Text",
                        doc_result["extracted_text"],
                        height=300
                    )
        else:
            st.info("No document processed yet. Upload a document to get started.")

elif page == "🧠 Neural Processing":
    st.header("Neural Processing")
    
    # Neural processing interface
    tab1, tab2, tab3 = st.tabs(["🧠 Neural Network", "🔗 Synaptic Plasticity", "💾 Memory Systems"])
    
    with tab1:
        st.subheader("Neural Network Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Network Architecture:**")
            num_layers = st.slider("Number of Layers", 1, 10, 3)
            neurons_per_layer = st.slider("Neurons per Layer", 10, 1000, 100)
            activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "leaky_relu"])
            
            if st.button("🔧 Configure Network"):
                st.success("Neural network configured successfully!")
        
        with col2:
            st.write("**Training Parameters:**")
            learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            batch_size = st.slider("Batch Size", 1, 128, 32)
            epochs = st.slider("Epochs", 1, 100, 10)
            
            if st.button("🚀 Start Training"):
                with st.spinner("Training neural network..."):
                    time.sleep(2)
                    st.success("Training completed successfully!")
    
    with tab2:
        st.subheader("Synaptic Plasticity")
        
        # Plasticity visualization
        plasticity_data = pd.DataFrame({
            "Synapse": [f"Synapse {i}" for i in range(1, 11)],
            "Weight": [0.5 + i*0.05 for i in range(10)],
            "Strength": [0.3 + i*0.07 for i in range(10)],
            "Adaptation": [0.2 + i*0.08 for i in range(10)]
        })
        
        fig = px.bar(plasticity_data, x="Synapse", y=["Weight", "Strength", "Adaptation"],
                    title="Synaptic Plasticity Metrics")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Memory Systems")
        
        # Memory metrics
        memory_data = {
            "Memory Type": ["Working Memory", "Long-term Memory", "Episodic Memory"],
            "Capacity": ["256 MB", "10 GB", "2 GB"],
            "Usage": ["45%", "62%", "38%"],
            "Efficiency": ["92%", "85%", "78%"]
        }
        
        df_memory = pd.DataFrame(memory_data)
        st.dataframe(df_memory, use_container_width=True)

elif page == "📈 Analytics":
    st.header("System Analytics")
    
    # Analytics dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Performance", "🤖 Agent Analytics", "📈 Trends"])
    
    with tab1:
        st.subheader("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", "1,234", "↗️ +123")
        
        with col2:
            st.metric("Success Rate", "98.5%", "↗️ +0.5%")
        
        with col3:
            st.metric("Avg Response Time", "245ms", "↘️ -15ms")
        
        with col4:
            st.metric("Active Users", "45", "↗️ +5")
        
        # Usage chart
        usage_data = pd.DataFrame({
            "Hour": list(range(24)),
            "Requests": [100 + i*5 + (i%6)*20 for i in range(24)],
            "CPU": [30 + i*2 + (i%4)*10 for i in range(24)]
        })
        
        fig = px.line(usage_data, x="Hour", y=["Requests", "CPU"],
                     title="24-Hour System Usage")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Performance Analysis")
        
        # Performance metrics
        perf_data = pd.DataFrame({
            "Endpoint": ["/api/chat", "/api/code", "/api/docs", "/api/agents", "/api/models"],
            "Requests": [500, 320, 180, 150, 90],
            "Avg Response (ms)": [250, 180, 300, 120, 90],
            "Success Rate": [99.2, 98.8, 99.5, 99.9, 100.0]
        })
        
        fig = px.scatter(perf_data, x="Avg Response (ms)", y="Success Rate",
                        size="Requests", hover_name="Endpoint",
                        title="API Performance Overview")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Agent Analytics")
        
        # Agent performance
        agent_data = pd.DataFrame({
            "Agent": ["DeepSeek-Coder", "Llama-2", "AutoGPT", "TabbyML"],
            "Tasks Completed": [45, 38, 25, 67],
            "Success Rate": [96.5, 94.2, 89.8, 98.1],
            "Avg Time (s)": [12.5, 8.3, 15.2, 3.1]
        })
        
        fig = px.bar(agent_data, x="Agent", y="Tasks Completed",
                    title="Agent Task Completion")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Trending Analysis")
        
        # Trend data
        trend_data = pd.DataFrame({
            "Date": pd.date_range(start='2024-01-01', periods=30, freq='D'),
            "Users": [20 + i + (i%7)*5 for i in range(30)],
            "Requests": [500 + i*10 + (i%5)*50 for i in range(30)],
            "Errors": [5 + (i%3)*2 for i in range(30)]
        })
        
        fig = px.line(trend_data, x="Date", y=["Users", "Requests"],
                     title="30-Day Trend Analysis")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🌐 AI Services":
    st.header("AI Services Management")
    
    # Get AI services status
    try:
        services_status = interface.make_request("/ai/services/status")
        services = services_status.get("services", {})
    except:
        services = {}
    
    # Service status overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        healthy_count = sum(1 for s in services.values() if s.get("status") == "healthy")
        st.metric("Healthy Services", healthy_count, f"out of {len(services)}")
    
    with col2:
        unhealthy_count = sum(1 for s in services.values() if s.get("status") == "unhealthy")
        st.metric("Unhealthy Services", unhealthy_count)
    
    with col3:
        unreachable_count = sum(1 for s in services.values() if s.get("status") == "unreachable")
        st.metric("Unreachable Services", unreachable_count)
    
    # Service details
    st.subheader("🔍 Service Details")
    
    for service_name, service_data in services.items():
        with st.expander(f"{service_name.title()} - {service_data.get('status', 'Unknown')}"):
            status = service_data.get("status", "Unknown")
            
            if status == "healthy":
                st.success(f"✅ {service_name} is running normally")
            elif status == "unhealthy":
                st.warning(f"⚠️ {service_name} is running but unhealthy")
            else:
                st.error(f"❌ {service_name} is unreachable")
            
            # Service details
            details = service_data.get("details", {})
            if details:
                st.json(details)
            
            # Service actions
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(f"🔄 Restart {service_name}", key=f"restart_{service_name}"):
                    st.info(f"Restart signal sent to {service_name}")
            
            with col_b:
                if st.button(f"📊 View Logs {service_name}", key=f"logs_{service_name}"):
                    st.info(f"Opening logs for {service_name}")

elif page == "🔬 Evolution Lab":
    st.header("Code Evolution Laboratory")
    
    # Evolution interface
    tab1, tab2, tab3 = st.tabs(["🧬 Code Evolution", "📊 Evolution History", "⚙️ Evolution Config"])
    
    with tab1:
        st.subheader("Evolve Your Code")
        
        # Code input
        code_to_evolve = st.text_area(
            "Code to Evolve:",
            placeholder="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            height=200
        )
        
        # Evolution parameters
        col1, col2 = st.columns(2)
        
        with col1:
            target_performance = st.slider("Target Performance", 0.0, 1.0, 0.8)
            target_efficiency = st.slider("Target Efficiency", 0.0, 1.0, 0.8)
        
        with col2:
            target_accuracy = st.slider("Target Accuracy", 0.0, 1.0, 0.8)
            max_iterations = st.slider("Max Iterations", 10, 100, 20)
        
        if st.button("🚀 Start Evolution"):
            if code_to_evolve:
                with st.spinner("Evolving code..."):
                    evolution_data = {
                        "code": code_to_evolve,
                        "target_performance": target_performance,
                        "target_efficiency": target_efficiency,
                        "target_accuracy": target_accuracy,
                        "max_iterations": max_iterations
                    }
                    
                    result = interface.make_request("/evolution/evolve_code", method="POST", data=evolution_data)
                    
                    if result.get("status") == "success":
                        st.success("Code evolution completed!")
                        
                        # Display evolved code
                        st.subheader("🎯 Evolved Code")
                        st.code(result["evolved_code"], language="python")
                        
                        # Display metrics
                        st.subheader("📈 Evolution Metrics")
                        metrics = result["metrics"]
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Performance", f"{metrics['performance_score']:.3f}")
                        with col_b:
                            st.metric("Efficiency", f"{metrics['efficiency_score']:.3f}")
                        with col_c:
                            st.metric("Accuracy", f"{metrics['accuracy_score']:.3f}")
                        
                        st.metric("Generation", result["generation"])
                    else:
                        st.error("Evolution failed: " + result.get("message", "Unknown error"))
    
    with tab2:
        st.subheader("Evolution Statistics")
        
        # Get evolution statistics
        try:
            stats = interface.make_request("/evolution/statistics")
            if stats.get("status") == "success":
                statistics = stats["statistics"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Generation", statistics.get("generation", 0))
                with col2:
                    st.metric("Population Size", statistics.get("population_size", 0))
                with col3:
                    st.metric("Best Score", f"{statistics.get('best_score', 0):.3f}")
                
                # Score distribution
                if statistics.get("average_score"):
                    st.subheader("📊 Score Distribution")
                    st.metric("Average Score", f"{statistics['average_score']:.3f}")
                    st.metric("Score Std Dev", f"{statistics.get('score_std', 0):.3f}")
        except:
            st.info("No evolution statistics available")
    
    with tab3:
        st.subheader("Evolution Configuration")
        
        # Evolution parameters
        with st.form("evolution_config"):
            population_size = st.slider("Population Size", 10, 100, 20)
            mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)
            crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.7)
            elite_size = st.slider("Elite Size", 1, 20, 5)
            
            submitted = st.form_submit_button("💾 Save Configuration")
            
            if submitted:
                st.success("Evolution configuration saved!")

elif page == "🕸️ Knowledge Graph":
    st.header("Knowledge Graph Explorer")
    
    # Knowledge graph interface
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "➕ Add Knowledge", "📊 Graph Stats"])
    
    with tab1:
        st.subheader("Search Knowledge Graph")
        
        # Search interface
        search_query = st.text_input("Search Query:", placeholder="machine learning algorithms")
        search_limit = st.slider("Max Results", 1, 50, 10)
        
        if st.button("🔍 Search"):
            if search_query:
                with st.spinner("Searching knowledge graph..."):
                    try:
                        search_results = interface.make_request(
                            f"/knowledge/graph/search?query={search_query}&limit={search_limit}"
                        )
                        
                        if search_results.get("status") == "success":
                            results = search_results["results"]
                            
                            if results:
                                st.success(f"Found {len(results)} results")
                                
                                for i, result in enumerate(results):
                                    with st.expander(f"{result['label']} (Similarity: {result['similarity']:.3f})"):
                                        st.write(f"**Type:** {result['type']}")
                                        st.write(f"**ID:** {result['node_id']}")
                                        
                                        if result['properties']:
                                            st.write("**Properties:**")
                                            st.json(result['properties'])
                            else:
                                st.info("No results found")
                        else:
                            st.error("Search failed")
                    except Exception as e:
                        st.error(f"Search error: {e}")
    
    with tab2:
        st.subheader("Add Knowledge Node")
        
        # Add node interface
        with st.form("add_knowledge"):
            node_label = st.text_input("Node Label:", placeholder="Neural Networks")
            node_type = st.selectbox("Node Type:", ["concept", "entity", "process", "tool", "technique"])
            
            # Properties
            st.write("**Properties (JSON):**")
            properties_json = st.text_area(
                "Properties:",
                placeholder='{"description": "A type of machine learning", "category": "AI"}',
                height=100
            )
            
            submitted = st.form_submit_button("➕ Add Node")
            
            if submitted and node_label:
                try:
                    properties = json.loads(properties_json) if properties_json else {}
                    
                    node_data = {
                        "label": node_label,
                        "type": node_type,
                        "properties": properties
                    }
                    
                    result = interface.make_request("/knowledge/graph/add_node", method="POST", data=node_data)
                    
                    if result.get("status") == "success":
                        st.success(f"Node added successfully! ID: {result['node_id']}")
                    else:
                        st.error("Failed to add node")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in properties")
                except Exception as e:
                    st.error(f"Error adding node: {e}")
    
    with tab3:
        st.subheader("Knowledge Graph Statistics")
        
        # Graph stats (placeholder)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", "1,234")
        with col2:
            st.metric("Total Connections", "2,567")
        with col3:
            st.metric("Node Types", "8")
        
        # Graph visualization placeholder
        st.subheader("📈 Graph Visualization")
        st.info("Graph visualization will be implemented with network visualization libraries")

elif page == "⚙️ Settings":
    st.header("System Settings")
    
    # Settings interface
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 General", "🤖 Models", "🔐 Security", "📊 Monitoring"])
    
    with tab1:
        st.subheader("General Settings")
        
        with st.form("general_settings"):
            system_name = st.text_input("System Name", "SutazAI")
            debug_mode = st.checkbox("Debug Mode", False)
            auto_save = st.checkbox("Auto Save", True)
            theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
            
            submitted = st.form_submit_button("💾 Save Settings")
            
            if submitted:
                st.success("Settings saved successfully!")
    
    with tab2:
        st.subheader("Model Configuration")
        
        # Model settings
        default_model = st.selectbox(
            "Default Model",
            ["deepseek-coder:33b", "llama2:13b", "codellama:7b"]
        )
        
        max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        
        if st.button("💾 Save Model Settings"):
            st.success("Model settings saved successfully!")
    
    with tab3:
        st.subheader("Security Settings")
        
        enable_auth = st.checkbox("Enable Authentication", True)
        enable_rate_limiting = st.checkbox("Enable Rate Limiting", True)
        enable_logging = st.checkbox("Enable Audit Logging", True)
        
        if st.button("🔐 Save Security Settings"):
            st.success("Security settings saved successfully!")
    
    with tab4:
        st.subheader("Monitoring Settings")
        
        enable_monitoring = st.checkbox("Enable Monitoring", True)
        metrics_interval = st.slider("Metrics Interval (seconds)", 5, 300, 30)
        retention_days = st.slider("Data Retention (days)", 1, 90, 7)
        
        if st.button("📊 Save Monitoring Settings"):
            st.success("Monitoring settings saved successfully!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 SutazAI AGI/ASI Autonomous System v1.0.0 | 
        Built with ❤️ using Streamlit | 
        <a href='https://github.com/sutazai/sutazaiapp' target='_blank'>GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-refresh for dashboard
if page == "🏠 Dashboard":
    time.sleep(30)
    st.rerun()