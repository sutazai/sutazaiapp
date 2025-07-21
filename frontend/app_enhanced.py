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

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .agent-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
    }
    .unhealthy {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {}
if 'system_metrics' not in st.session_state:
    st.session_state.system_metrics = {}

# API configuration
API_BASE_URL = "http://localhost:8000"

async def call_api(endpoint: str, method: str = "GET", data: Dict = None):
    """Call backend API"""
    async with httpx.AsyncClient() as client:
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
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None

def main():
    """Main application"""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🧠 SutazAI AGI/ASI System")
        st.caption("Autonomous General Intelligence Platform")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100?text=SutazAI+Logo", use_column_width=True)
        st.markdown("---")
        
        # System Status
        status = asyncio.run(call_api("/health"))
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
            "💡 Knowledge Base",
            "📊 Analytics",
            "🔧 System Config",
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
    elif page == "💡 Knowledge Base":
        show_knowledge_base()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "🔧 System Config":
        show_system_config()
    elif page == "🚀 Self-Improvement":
        show_self_improvement()

def show_dashboard():
    """Show main dashboard"""
    st.header("System Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Agents", "22", "+3")
    with col2:
        st.metric("Tasks Completed", "1,247", "+58")
    with col3:
        st.metric("Knowledge Entries", "15.2K", "+342")
    with col4:
        st.metric("System Health", "98%", "+2%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent activity chart
        st.subheader("Agent Activity")
        
        # Sample data
        agent_data = pd.DataFrame({
            'Agent': ['AutoGPT', 'CrewAI', 'GPT-Engineer', 'Aider', 'BigAGI'],
            'Tasks': [234, 189, 156, 142, 98]
        })
        
        fig = px.bar(agent_data, x='Agent', y='Tasks', 
                     title="Tasks by Agent (Last 24h)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # System performance
        st.subheader("System Performance")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 92,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Performance"},
            delta = {'reference': 85},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent System Activity")
    
    activities = [
        {"time": "2 min ago", "event": "AutoGPT completed task: Market analysis", "type": "success"},
        {"time": "5 min ago", "event": "Knowledge base updated with 45 new entries", "type": "info"},
        {"time": "12 min ago", "event": "Self-improvement: Code optimization applied", "type": "warning"},
        {"time": "18 min ago", "event": "CrewAI team completed collaborative task", "type": "success"},
        {"time": "25 min ago", "event": "System backup completed successfully", "type": "info"}
    ]
    
    for activity in activities:
        if activity["type"] == "success":
            st.success(f"🟢 {activity['time']}: {activity['event']}")
        elif activity["type"] == "warning":
            st.warning(f"🟡 {activity['time']}: {activity['event']}")
        else:
            st.info(f"🔵 {activity['time']}: {activity['event']}")

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
    
    # Input
    if prompt := st.chat_input("Ask anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("Thinking..."):
            if model == "AGI Brain (Multi-Model)":
                response = asyncio.run(call_api("/think", "POST", {"query": prompt}))
            else:
                response = asyncio.run(call_api("/chat", "POST", {
                    "message": prompt,
                    "model": model.lower().replace(" ", "-"),
                    "agent": agent if agent != "None (Direct Model)" else None
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
    agents = asyncio.run(call_api("/agents"))
    
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
                agent = next((a for a in agents if a["name"] == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
        
        with tab2:
            st.subheader("Code Generation Agents")
            col1, col2 = st.columns(2)
            
            code_agents = ["GPT-Engineer", "Aider", "TabbyML", "Semgrep"]
            for i, agent_name in enumerate(code_agents):
                agent = next((a for a in agents if a["name"] == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
        
        with tab3:
            st.subheader("Web Automation Agents")
            col1, col2 = st.columns(2)
            
            web_agents = ["BrowserUse", "Skyvern", "AgentGPT"]
            for i, agent_name in enumerate(web_agents):
                agent = next((a for a in agents if a["name"] == agent_name), None)
                if agent:
                    with col1 if i % 2 == 0 else col2:
                        show_agent_card(agent)
        
        with tab4:
            st.subheader("Specialized Agents")
            col1, col2 = st.columns(2)
            
            special_agents = ["Documind", "FinRobot", "BigAGI", "AgentZero"]
            for i, agent_name in enumerate(special_agents):
                agent = next((a for a in agents if a["name"] == agent_name), None)
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
            st.text_input("Version:", value="3.0.0", disabled=True)
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

if __name__ == "__main__":
    main() 