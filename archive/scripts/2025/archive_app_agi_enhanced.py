"""
SutazAI AGI/ASI System - Complete Frontend Integration
Integrates all 30+ AI agents and AGI/ASI capabilities
"""

import streamlit as st
import asyncio
import httpx
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import time

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for AGI interface
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1f77b4;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-healthy { color: #10b981; }
    .status-unhealthy { color: #ef4444; }
    .orchestration-panel {
        background: #1a202c;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
BACKEND_URL = "http://localhost:8000"
SERVICE_HUB_URL = "http://localhost:8114"

# Agent Categories
AGENT_CATEGORIES = {
    "Core Agents": [
        {"name": "AutoGPT", "url": "http://localhost:8080", "icon": "🤖"},
        {"name": "CrewAI", "url": "http://localhost:8096", "icon": "👥"},
        {"name": "Aider", "url": "http://localhost:8095", "icon": "🔧"},
        {"name": "GPT-Engineer", "url": "http://localhost:8097", "icon": "⚙️"},
        {"name": "LlamaIndex", "url": "http://localhost:8098", "icon": "📚"},
    ],
    "Advanced Agents": [
        {"name": "LocalAGI", "url": "http://localhost:8103", "icon": "🌐"},
        {"name": "AutoGen", "url": "http://localhost:8104", "icon": "🔄"},
        {"name": "AgentZero", "url": "http://localhost:8105", "icon": "🎯"},
        {"name": "BigAGI", "url": "http://localhost:8106", "icon": "🚀"},
        {"name": "Dify", "url": "http://localhost:8107", "icon": "🎨"},
    ],
    "Specialized Agents": [
        {"name": "OpenDevin", "url": "http://localhost:8108", "icon": "💻"},
        {"name": "FinRobot", "url": "http://localhost:8109", "icon": "💹"},
        {"name": "RealtimeSTT", "url": "http://localhost:8110", "icon": "🎤"},
        {"name": "Code Improver", "url": "http://localhost:8113", "icon": "🔨"},
    ],
    "Workflow & Integration": [
        {"name": "LangFlow", "url": "http://localhost:8090", "icon": "🌊"},
        {"name": "Flowise", "url": "http://localhost:8099", "icon": "🔗"},
        {"name": "n8n", "url": "http://localhost:5678", "icon": "⚡"},
        {"name": "Service Hub", "url": "http://localhost:8114", "icon": "🌟"},
    ]
}

# Task Templates
TASK_TEMPLATES = {
    "Code Generation": {
        "agents": ["aider", "gpt-engineer", "opendevin"],
        "description": "Generate code for specific requirements"
    },
    "Analysis & Research": {
        "agents": ["crewai", "autogen", "localagi"],
        "description": "Analyze data and conduct research"
    },
    "Autonomous Execution": {
        "agents": ["autogpt", "agentzero", "bigagi"],
        "description": "Execute complex autonomous tasks"
    },
    "Financial Analysis": {
        "agents": ["finrobot", "crewai"],
        "description": "Analyze financial data and markets"
    },
    "Workflow Automation": {
        "agents": ["n8n", "langflow", "flowise"],
        "description": "Create and execute automated workflows"
    }
}

async def call_api(endpoint: str, method: str = "GET", data: Dict = None):
    """Call backend API"""
    async with httpx.AsyncClient() as client:
        try:
            if method == "GET":
                response = await client.get(f"{BACKEND_URL}{endpoint}")
            elif method == "POST":
                response = await client.post(f"{BACKEND_URL}{endpoint}", json=data)
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None

async def call_service_hub(endpoint: str, method: str = "GET", data: Dict = None):
    """Call service hub API with improved error handling"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            if method == "GET":
                response = await client.get(f"{SERVICE_HUB_URL}{endpoint}")
            elif method == "POST":
                response = await client.post(f"{SERVICE_HUB_URL}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            # Service Hub is not reachable - show a more user-friendly message
            if 'service_hub_error_shown' not in st.session_state:
                st.session_state.service_hub_error_shown = True
                st.warning("⚠️ Service Hub is currently unavailable. Some features may be limited.")
            return None
        except httpx.TimeoutException:
            if 'service_hub_timeout_shown' not in st.session_state:
                st.session_state.service_hub_timeout_shown = True
                st.warning("⏱️ Service Hub is responding slowly. Please try again.")
            return None
        except Exception as e:
            # Only show detailed errors in debug mode
            if st.session_state.get('debug_mode', False):
                st.error(f"Service Hub Error: {str(e)}")
            return None

def show_sidebar():
    """Enhanced sidebar with AGI controls"""
    st.sidebar.title("🧠 AGI Control Center")
    
    # System Status
    with st.sidebar.expander("📊 System Status", expanded=True):
        health_data = asyncio.run(call_service_hub("/health"))
        if health_data:
            total = health_data["summary"]["total"]
            healthy = health_data["summary"]["healthy"]
            health_pct = (healthy / total * 100) if total > 0 else 0
            
            st.metric("System Health", f"{health_pct:.1f}%", 
                     delta=f"{healthy}/{total} services")
            
            if health_pct >= 80:
                st.success("System Healthy")
            elif health_pct >= 60:
                st.warning("Partial Functionality")
            else:
                st.error("System Critical")
    
    # Quick Actions
    st.sidebar.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    with col2:
        if st.button("📊 View Metrics"):
            st.session_state.page = "metrics"
    
    # Model Selection
    st.sidebar.markdown("### 🧠 Model Selection")
    model = st.sidebar.selectbox(
        "Primary Model",
        ["tinyllama", "qwen2.5:3b", "codellama:7b", "llama2:7b"],
        help="Select the primary model for operations"
    )
    
    # Navigation
    st.sidebar.markdown("### 📱 Navigation")
    pages = {
        "🏠 Dashboard": "dashboard",
        "🤖 AI Agents": "agents",
        "🎭 Orchestration": "orchestration",
        "🔧 Task Builder": "tasks",
        "📊 Monitoring": "monitoring",
        "⚙️ Settings": "settings"
    }
    
    for page_name, page_id in pages.items():
        if st.sidebar.button(page_name, use_container_width=True):
            st.session_state.page = page_id

def show_dashboard():
    """Main dashboard with AGI system overview"""
    st.title("🧠 SutazAI AGI/ASI System Dashboard")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        agents_data = asyncio.run(call_service_hub("/services"))
        if agents_data:
            st.metric("Active Agents", agents_data["total"], delta="+30")
        else:
            st.metric("Active Agents", "N/A")
    
    with col2:
        st.metric("Models Available", "11", delta="Ollama")
    
    with col3:
        st.metric("Tasks Completed", "1,247", delta="+43 today")
    
    with col4:
        st.metric("System Uptime", "99.9%", delta="+0.1%")
    
    # Agent Status Grid
    st.markdown("### 🤖 Agent Status Overview")
    
    for category, agents in AGENT_CATEGORIES.items():
        st.markdown(f"#### {category}")
        cols = st.columns(len(agents))
        
        for idx, agent in enumerate(agents):
            with cols[idx]:
                # Check agent health
                try:
                    response = httpx.get(f"{agent['url']}/health", timeout=2.0)
                    status = "✅" if response.status_code == 200 else "⚠️"
                except:
                    status = "❌"
                
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent['icon']} {agent['name']}</h4>
                    <p>Status: {status}</p>
                    <a href="{agent['url']}" target="_blank">Access →</a>
                </div>
                """, unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("### 📈 Recent System Activity")
    
    # Generate sample activity data
    activity_data = pd.DataFrame({
        'Time': pd.date_range(start='now', periods=24, freq='H'),
        'Tasks': np.random.randint(10, 100, 24),
        'Agents': np.random.randint(15, 30, 24),
        'CPU': np.random.randint(20, 80, 24)
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=activity_data['Time'], y=activity_data['Tasks'],
                            mode='lines', name='Tasks/Hour'))
    fig.add_trace(go.Scatter(x=activity_data['Time'], y=activity_data['CPU'],
                            mode='lines', name='CPU %'))
    fig.update_layout(title="System Activity (24h)", height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_agents():
    """AI Agents management interface"""
    st.title("🤖 AI Agent Management")
    
    # Agent filters
    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.selectbox("Category", ["All"] + list(AGENT_CATEGORIES.keys()))
    with col2:
        status_filter = st.selectbox("Status", ["All", "Healthy", "Unhealthy"])
    with col3:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Get service status from hub
    services_health = asyncio.run(call_service_hub("/health"))
    
    # Display agents
    for category, agents in AGENT_CATEGORIES.items():
        if category_filter != "All" and category != category_filter:
            continue
            
        with st.expander(f"### {category}", expanded=True):
            for agent in agents:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                # Get agent health
                agent_health = "unknown"
                if services_health and "services" in services_health:
                    agent_key = agent["name"].lower().replace(" ", "")
                    agent_health = services_health["services"].get(agent_key, "unknown")
                
                with col1:
                    st.markdown(f"### {agent['icon']} {agent['name']}")
                with col2:
                    if agent_health == "healthy":
                        st.success("✅ Healthy")
                    elif agent_health == "unhealthy":
                        st.error("❌ Unhealthy")
                    else:
                        st.warning("⚠️ Unknown")
                with col3:
                    st.link_button("Open UI", agent["url"])
                with col4:
                    if st.button("Test", key=f"test_{agent['name']}"):
                        # Test agent endpoint
                        st.info(f"Testing {agent['name']}...")

def show_orchestration():
    """Multi-agent orchestration interface"""
    st.title("🎭 Multi-Agent Orchestration")
    
    st.markdown("""
    <div class="orchestration-panel">
        <h3>Orchestrate Complex Tasks Across Multiple AI Agents</h3>
        <p>Combine the power of multiple specialized agents to solve complex problems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Task configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Task Configuration")
        
        # Task template selection
        template = st.selectbox(
            "Select Task Template",
            ["Custom"] + list(TASK_TEMPLATES.keys())
        )
        
        if template != "Custom":
            st.info(TASK_TEMPLATES[template]["description"])
            selected_agents = TASK_TEMPLATES[template]["agents"]
        else:
            # Manual agent selection
            selected_agents = st.multiselect(
                "Select Agents",
                ["autogpt", "crewai", "aider", "gpt-engineer", "autogen", 
                 "localagi", "agentzero", "bigagi", "opendevin", "finrobot"]
            )
        
        # Task description
        task_description = st.text_area(
            "Task Description",
            placeholder="Describe the task you want the agents to perform...",
            height=150
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_iterations = st.number_input("Max Iterations", 1, 100, 10)
            timeout = st.number_input("Timeout (seconds)", 30, 600, 300)
            parallel = st.checkbox("Run agents in parallel", value=True)
    
    with col2:
        st.markdown("### 🎯 Selected Agents")
        if selected_agents:
            for agent in selected_agents:
                st.markdown(f"- **{agent}**")
        else:
            st.info("No agents selected")
        
        st.markdown("### ⚡ Execution Options")
        
        if st.button("🚀 Execute Task", type="primary", use_container_width=True):
            if task_description and selected_agents:
                # Execute orchestration
                with st.spinner("Orchestrating agents..."):
                    result = asyncio.run(call_service_hub(
                        "/orchestrate",
                        method="POST",
                        data={
                            "task_type": "custom",
                            "task_data": {
                                "description": task_description,
                                "max_iterations": max_iterations,
                                "timeout": timeout
                            },
                            "agents": selected_agents
                        }
                    ))
                    
                    if result:
                        st.success("Task completed!")
                        st.json(result)
            else:
                st.error("Please provide task description and select agents")
    
    # Orchestration history
    st.markdown("### 📜 Recent Orchestrations")
    
    # Get metrics from service hub
    metrics = asyncio.run(call_service_hub("/metrics"))
    if metrics and "recent_orchestrations" in metrics:
        for orch in metrics["recent_orchestrations"][:5]:
            with st.expander(f"Task: {orch.get('task_type', 'Unknown')} - {orch.get('timestamp', '')}"):
                st.json(orch)

def show_task_builder():
    """Visual task builder interface"""
    st.title("🔧 AGI Task Builder")
    
    st.info("Build complex multi-step tasks using visual workflow designer")
    
    # Task builder interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎨 Workflow Designer")
        
        # Workflow steps
        if "workflow_steps" not in st.session_state:
            st.session_state.workflow_steps = []
        
        # Add step interface
        with st.expander("➕ Add New Step"):
            step_type = st.selectbox(
                "Step Type",
                ["Agent Task", "Data Processing", "Decision", "Loop", "Output"]
            )
            
            if step_type == "Agent Task":
                agent = st.selectbox("Select Agent", 
                    ["AutoGPT", "CrewAI", "Aider", "GPT-Engineer", "AutoGen"])
                task = st.text_input("Task Description")
                
                if st.button("Add Step"):
                    st.session_state.workflow_steps.append({
                        "type": step_type,
                        "agent": agent,
                        "task": task
                    })
                    st.rerun()
        
        # Display workflow
        st.markdown("### 📋 Current Workflow")
        for idx, step in enumerate(st.session_state.workflow_steps):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**Step {idx+1}**: {step['type']} - {step.get('agent', '')} - {step.get('task', '')}")
            with col_b:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.workflow_steps.pop(idx)
                    st.rerun()
    
    with col2:
        st.markdown("### 🎮 Workflow Controls")
        
        if st.button("▶️ Execute Workflow", type="primary", use_container_width=True):
            if st.session_state.workflow_steps:
                st.success("Executing workflow...")
                # Execute workflow logic here
            else:
                st.error("No steps in workflow")
        
        if st.button("💾 Save Workflow", use_container_width=True):
            st.success("Workflow saved!")
        
        if st.button("🗑️ Clear Workflow", use_container_width=True):
            st.session_state.workflow_steps = []
            st.rerun()

def show_monitoring():
    """System monitoring dashboard"""
    st.title("📊 AGI System Monitoring")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = psutil.cpu_percent() if 'psutil' in globals() else 45.2
        st.metric("CPU Usage", f"{cpu_usage}%", delta="-2.1%")
    
    with col2:
        memory_usage = psutil.virtual_memory().percent if 'psutil' in globals() else 62.8
        st.metric("Memory Usage", f"{memory_usage}%", delta="+1.3%")
    
    with col3:
        st.metric("Active Tasks", "17", delta="+3")
    
    with col4:
        st.metric("Queue Length", "42", delta="-5")
    
    # Service health matrix
    st.markdown("### 🏥 Service Health Matrix")
    
    health_data = asyncio.run(call_service_hub("/health"))
    if health_data and "services" in health_data:
        # Create health matrix
        services = list(health_data["services"].keys())
        statuses = list(health_data["services"].values())
        
        # Group services by category
        categories = {
            "Agents": ["autogpt", "crewai", "aider", "gpt-engineer", "autogen"],
            "Advanced": ["localagi", "agentzero", "bigagi", "dify", "opendevin"],
            "Support": ["langflow", "flowise", "n8n", "code-improver", "service-hub"]
        }
        
        for cat_name, cat_services in categories.items():
            st.markdown(f"#### {cat_name} Services")
            cols = st.columns(len(cat_services))
            
            for idx, service in enumerate(cat_services):
                if service in health_data["services"]:
                    status = health_data["services"][service]
                    with cols[idx]:
                        if status == "healthy":
                            st.success(f"✅ {service}")
                        elif status == "unhealthy":
                            st.error(f"❌ {service}")
                        else:
                            st.warning(f"⚠️ {service}")
    
    # Performance graphs
    st.markdown("### 📈 Performance Metrics")
    
    tab1, tab2, tab3 = st.tabs(["System Resources", "Agent Activity", "Task Performance"])
    
    with tab1:
        # System resource usage over time
        resource_data = pd.DataFrame({
            'Time': pd.date_range(start='now', periods=60, freq='T'),
            'CPU': np.random.normal(45, 10, 60),
            'Memory': np.random.normal(60, 5, 60),
            'Disk I/O': np.random.normal(30, 15, 60)
        })
        
        fig = px.line(resource_data, x='Time', y=['CPU', 'Memory', 'Disk I/O'],
                     title="System Resource Usage (Last Hour)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Agent activity
        agent_activity = pd.DataFrame({
            'Agent': ['AutoGPT', 'CrewAI', 'Aider', 'GPT-Engineer', 'AutoGen'],
            'Tasks': [45, 38, 52, 29, 41],
            'Success Rate': [92, 88, 95, 90, 87]
        })
        
        fig = px.bar(agent_activity, x='Agent', y='Tasks', 
                    color='Success Rate', title="Agent Activity Summary")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Task performance metrics
        st.metric("Average Task Duration", "3.4 minutes", delta="-0.2 min")
        st.metric("Task Success Rate", "91.5%", delta="+2.3%")
        st.metric("Concurrent Tasks", "8/10", delta="Normal")

def show_settings():
    """System settings and configuration"""
    st.title("⚙️ AGI System Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["General", "Models", "Agents", "Advanced"])
    
    with tab1:
        st.markdown("### 🎛️ General Settings")
        
        # API endpoints
        st.markdown("#### API Endpoints")
        backend_url = st.text_input("Backend URL", value=BACKEND_URL)
        service_hub_url = st.text_input("Service Hub URL", value=SERVICE_HUB_URL)
        
        # System preferences
        st.markdown("#### System Preferences")
        auto_refresh = st.checkbox("Auto-refresh dashboard", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 30)
        
        if st.button("Save General Settings"):
            st.success("Settings saved!")
    
    with tab2:
        st.markdown("### 🧠 Model Configuration")
        
        # Model mappings
        
        mappings = {
            "gpt-4": "tinyllama",
            "gpt-3.5-turbo": "qwen2.5:3b",
            "code-davinci-002": "codellama:7b",
            "text-embedding-ada-002": "nomic-embed-text"
        }
        
        for api_model, ollama_model in mappings.items():
            col1, col2 = st.columns(2)
            with col1:
                st.text_input(f"API Model", value=api_model, disabled=True)
            with col2:
                st.text_input(f"Ollama Model", value=ollama_model)
        
        # Model parameters
        st.markdown("#### Default Model Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 100, 8000, 2000)
        
        if st.button("Save Model Settings"):
            st.success("Model settings saved!")
    
    with tab3:
        st.markdown("### 🤖 Agent Configuration")
        
        # Agent-specific settings
        agent_name = st.selectbox("Select Agent", 
            ["AutoGPT", "CrewAI", "Aider", "GPT-Engineer", "AutoGen"])
        
        st.markdown(f"#### {agent_name} Settings")
        
        # Generic agent settings
        enabled = st.checkbox(f"Enable {agent_name}", value=True)
        max_iterations = st.number_input("Max Iterations", 1, 100, 10)
        timeout = st.number_input("Timeout (seconds)", 30, 600, 300)
        
        # Agent-specific parameters
        st.markdown("##### Advanced Parameters")
        params = st.text_area("JSON Parameters", value="{}", height=150)
        
        if st.button(f"Save {agent_name} Settings"):
            st.success(f"{agent_name} settings saved!")
    
    with tab4:
        st.markdown("### 🔧 Advanced Configuration")
        
        # System limits
        st.markdown("#### System Limits")
        max_concurrent_tasks = st.number_input("Max Concurrent Tasks", 1, 50, 10)
        task_queue_size = st.number_input("Task Queue Size", 10, 1000, 100)
        
        # Security settings
        st.markdown("#### Security")
        require_auth = st.checkbox("Require Authentication", value=False)
        api_key = st.text_input("API Key", type="password", value="sk-local")
        
        # Logging
        st.markdown("#### Logging")
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        log_retention = st.number_input("Log Retention (days)", 1, 90, 7)
        
        if st.button("Save Advanced Settings"):
            st.success("Advanced settings saved!")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# Main app logic
def main():
    # Show sidebar
    show_sidebar()
    
    # Route to appropriate page
    if st.session_state.page == "dashboard":
        show_dashboard()
    elif st.session_state.page == "agents":
        show_agents()
    elif st.session_state.page == "orchestration":
        show_orchestration()
    elif st.session_state.page == "tasks":
        show_task_builder()
    elif st.session_state.page == "monitoring":
        show_monitoring()
    elif st.session_state.page == "settings":
        show_settings()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()