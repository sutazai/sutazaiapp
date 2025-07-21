#!/usr/bin/env python3
"""
Orchestrated Frontend for SutazAI v9 Enterprise
Advanced Streamlit interface for multi-agent communication and orchestration
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import asyncio
import websockets
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Page config
st.set_page_config(
    page_title="SutazAI v9 Enterprise Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
ORCHESTRATOR_API = "http://localhost:9000"
BACKEND_API = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .healthy { border-left: 5px solid #28a745; }
    .unhealthy { border-left: 5px solid #dc3545; }
    .starting { border-left: 5px solid #ffc107; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{ORCHESTRATOR_API}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

def get_agents_data() -> List[Dict]:
    """Get all agents data"""
    result = make_api_request("/api/agents")
    if "error" in result:
        st.error(f"Failed to get agents: {result['error']}")
        return []
    return result

def get_system_status() -> Dict:
    """Get system status"""
    result = make_api_request("/api/system/status")
    if "error" in result:
        st.error(f"Failed to get system status: {result['error']}")
        return {}
    return result

def submit_task(description: str, capabilities: List[str], priority: int = 5) -> str:
    """Submit a task to the orchestrator"""
    data = {
        "description": description,
        "required_capabilities": capabilities,
        "priority": priority
    }
    result = make_api_request("/api/tasks", method="POST", data=data)
    if "error" in result:
        st.error(f"Failed to submit task: {result['error']}")
        return ""
    return result.get("task_id", "")

def get_task_status(task_id: str) -> Dict:
    """Get task status"""
    result = make_api_request(f"/api/tasks/{task_id}")
    if "error" in result:
        return {"error": result["error"]}
    return result

def chat_with_agent(agent_name: str, message: Dict) -> Dict:
    """Chat with a specific agent"""
    data = {
        "agent_name": agent_name,
        "message": message
    }
    result = make_api_request("/api/chat", method="POST", data=data)
    return result

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 SutazAI v9 Enterprise Orchestrator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "🤖 Agent Management", 
        "📋 Task Management",
        "💬 Multi-Agent Chat",
        "🔧 Workflows",
        "📊 Analytics",
        "⚙️ System Admin"
    ])
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Page routing
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🤖 Agent Management":
        agent_management_page()
    elif page == "📋 Task Management":
        task_management_page()
    elif page == "💬 Multi-Agent Chat":
        multi_agent_chat_page()
    elif page == "🔧 Workflows":
        workflows_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ System Admin":
        system_admin_page()

def dashboard_page():
    """Main dashboard page"""
    st.header("System Overview")
    
    # Get system status
    status = get_system_status()
    if not status:
        st.error("Unable to connect to orchestrator")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", status.get("total_agents", 0))
    
    with col2:
        healthy = status.get("healthy_agents", 0)
        st.metric("Healthy Agents", healthy, 
                 delta=f"{status.get('agent_health_rate', 0)*100:.1f}% health rate")
    
    with col3:
        st.metric("Total Tasks", status.get("total_tasks", 0))
    
    with col4:
        completed = status.get("completed_tasks", 0)
        st.metric("Completed Tasks", completed,
                 delta=f"{status.get('task_completion_rate', 0)*100:.1f}% completion rate")
    
    # Agent status overview
    st.subheader("Agent Status Overview")
    agents = status.get("agents", [])
    
    if agents:
        # Create agent status chart
        status_counts = {}
        for agent in agents:
            status_val = agent.get("status", "unknown")
            status_counts[status_val] = status_counts.get(status_val, 0) + 1
        
        fig = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Agent Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Agent grid
        cols = st.columns(3)
        for i, agent in enumerate(agents[:6]):  # Show first 6 agents
            with cols[i % 3]:
                status_class = agent.get("status", "unknown").lower()
                st.markdown(f"""
                <div class="agent-card {status_class}">
                    <h4>{agent.get("name", "Unknown")}</h4>
                    <p>Status: {agent.get("status", "Unknown")}</p>
                    <p>Port: {agent.get("port", "N/A")}</p>
                    <p>Success: {agent.get("success_count", 0)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Health Check All Agents"):
            with st.spinner("Checking agent health..."):
                # This would trigger health checks
                st.success("Health check initiated")
    
    with col2:
        if st.button("📝 Submit Test Task"):
            task_id = submit_task("Test task from dashboard", ["conversation"], 5)
            if task_id:
                st.success(f"Test task submitted: {task_id}")
    
    with col3:
        if st.button("📊 Generate Report"):
            st.info("Report generation feature coming soon")

def agent_management_page():
    """Agent management page"""
    st.header("Agent Management")
    
    # Get agents
    agents = get_agents_data()
    
    if not agents:
        st.warning("No agents found or unable to connect")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by status", 
                                   ["All"] + list(set(a.get("status", "") for a in agents)))
    
    with col2:
        capability_filter = st.selectbox("Filter by capability", 
                                       ["All"] + list(set(cap for a in agents for cap in a.get("capabilities", []))))
    
    # Filter agents
    filtered_agents = agents
    if status_filter != "All":
        filtered_agents = [a for a in filtered_agents if a.get("status") == status_filter]
    if capability_filter != "All":
        filtered_agents = [a for a in filtered_agents if capability_filter in a.get("capabilities", [])]
    
    # Agents table
    if filtered_agents:
        df = pd.DataFrame(filtered_agents)
        
        # Format capabilities for display
        if "capabilities" in df.columns:
            df["capabilities"] = df["capabilities"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        
        st.dataframe(df, use_container_width=True)
        
        # Agent details
        st.subheader("Agent Details")
        selected_agent = st.selectbox("Select agent for details", 
                                    [a.get("name", "") for a in filtered_agents])
        
        if selected_agent:
            agent = next((a for a in filtered_agents if a.get("name") == selected_agent), None)
            if agent:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(agent)
                
                with col2:
                    st.subheader("Agent Actions")
                    
                    if st.button(f"Test Chat with {selected_agent}"):
                        with st.spinner("Testing chat..."):
                            result = chat_with_agent(selected_agent, {"message": "Hello, are you working?"})
                            st.json(result)
                    
                    if st.button(f"Restart {selected_agent}"):
                        result = make_api_request(f"/api/admin/restart-agent/{selected_agent}", method="POST")
                        if "error" not in result:
                            st.success(f"Restart initiated for {selected_agent}")
                        else:
                            st.error(f"Failed to restart: {result['error']}")
                    
                    if st.button(f"View Logs for {selected_agent}"):
                        result = make_api_request(f"/api/admin/logs/{selected_agent}")
                        if "error" not in result:
                            st.text_area("Logs", result.get("logs", ""), height=300)
                        else:
                            st.error(f"Failed to get logs: {result['error']}")

def task_management_page():
    """Task management page"""
    st.header("Task Management")
    
    # Submit new task
    st.subheader("Submit New Task")
    
    with st.form("submit_task"):
        task_description = st.text_area("Task Description", 
                                      placeholder="Describe what you want the AI agents to do...")
        
        # Capability selection
        available_capabilities = [
            "code_generation", "document_processing", "web_automation",
            "conversation", "reasoning", "multi_agent_coordination",
            "research", "analysis", "planning", "financial_analysis",
            "knowledge_management"
        ]
        
        selected_capabilities = st.multiselect("Required Capabilities", available_capabilities)
        priority = st.slider("Priority", 1, 10, 5)
        
        submitted = st.form_submit_button("Submit Task")
        
        if submitted and task_description and selected_capabilities:
            task_id = submit_task(task_description, selected_capabilities, priority)
            if task_id:
                st.success(f"Task submitted successfully! ID: {task_id}")
                st.session_state.last_task_id = task_id
    
    # Task status tracking
    st.subheader("Task Status Tracking")
    
    if hasattr(st.session_state, 'last_task_id'):
        task_id = st.session_state.last_task_id
        
        if st.button("Check Task Status"):
            task_status = get_task_status(task_id)
            if "error" not in task_status:
                st.json(task_status)
                
                # Progress indicator
                status = task_status.get("status", "unknown")
                if status == "completed":
                    st.success("✅ Task completed!")
                elif status == "in_progress":
                    st.info("🔄 Task in progress...")
                elif status == "failed":
                    st.error("❌ Task failed")
                elif status == "pending":
                    st.warning("⏳ Task pending...")
            else:
                st.error(f"Error: {task_status['error']}")
    
    # Manual task ID input
    manual_task_id = st.text_input("Or enter Task ID manually:")
    if st.button("Check Manual Task ID") and manual_task_id:
        task_status = get_task_status(manual_task_id)
        if "error" not in task_status:
            st.json(task_status)
        else:
            st.error(f"Error: {task_status['error']}")

def multi_agent_chat_page():
    """Multi-agent chat interface"""
    st.header("Multi-Agent Chat Interface")
    
    # Get available agents
    agents = get_agents_data()
    healthy_agents = [a for a in agents if a.get("status") == "healthy"]
    
    if not healthy_agents:
        st.warning("No healthy agents available for chat")
        return
    
    # Agent selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_agent = st.selectbox("Select Agent", 
                                    [a.get("name", "") for a in healthy_agents])
    
    with col2:
        chat_mode = st.selectbox("Chat Mode", ["Single Agent", "Broadcast to All"])
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your message..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Send to agent(s)
        if chat_mode == "Single Agent" and selected_agent:
            with st.chat_message("assistant"):
                with st.spinner(f"Sending to {selected_agent}..."):
                    result = chat_with_agent(selected_agent, {"message": prompt})
                    
                    if "error" not in result:
                        response = result.get("response", "No response received")
                        st.write(f"**{selected_agent}**: {response}")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**{selected_agent}**: {response}"
                        })
                    else:
                        st.error(f"Error from {selected_agent}: {result['error']}")
        
        elif chat_mode == "Broadcast to All":
            with st.chat_message("assistant"):
                with st.spinner("Broadcasting to all agents..."):
                    result = make_api_request("/api/broadcast", method="POST", 
                                            data={"message": prompt})
                    
                    if "error" not in result:
                        results = result.get("results", {})
                        responses = []
                        
                        for agent_name, agent_result in results.items():
                            if "error" not in agent_result:
                                response = agent_result.get("response", "No response")
                                responses.append(f"**{agent_name}**: {response}")
                            else:
                                responses.append(f"**{agent_name}**: Error - {agent_result['error']}")
                        
                        full_response = "\n\n".join(responses)
                        st.write(full_response)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_response
                        })
                    else:
                        st.error(f"Broadcast error: {result['error']}")
    
    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def workflows_page():
    """Pre-built workflows page"""
    st.header("Multi-Agent Workflows")
    
    st.subheader("Code Generation Workflow")
    with st.form("code_workflow"):
        code_prompt = st.text_area("Code Requirements", 
                                 placeholder="Describe the code you want to generate...")
        code_language = st.selectbox("Programming Language", 
                                   ["python", "javascript", "java", "go", "rust"])
        
        if st.form_submit_button("Generate Code"):
            if code_prompt:
                with st.spinner("Executing code generation workflow..."):
                    result = make_api_request("/api/workflows/code-generation", 
                                            method="POST",
                                            data={"prompt": code_prompt, "language": code_language})
                    
                    if "error" not in result:
                        st.success("Workflow initiated!")
                        st.json(result)
                    else:
                        st.error(f"Workflow error: {result['error']}")
    
    st.subheader("Research & Analysis Workflow")
    with st.form("research_workflow"):
        research_topic = st.text_input("Research Topic", 
                                     placeholder="Enter topic to research...")
        
        if st.form_submit_button("Start Research"):
            if research_topic:
                with st.spinner("Executing research workflow..."):
                    result = make_api_request("/api/workflows/research-analysis",
                                            method="POST", 
                                            data={"topic": research_topic})
                    
                    if "error" not in result:
                        st.success("Research workflow initiated!")
                        st.json(result)
                    else:
                        st.error(f"Workflow error: {result['error']}")

def analytics_page():
    """Analytics and monitoring page"""
    st.header("System Analytics")
    
    # Get system status for analytics
    status = get_system_status()
    agents = status.get("agents", [])
    
    if not agents:
        st.warning("No data available for analytics")
        return
    
    # Performance metrics
    st.subheader("Agent Performance Metrics")
    
    # Response time chart
    response_times = [(a.get("name", ""), a.get("response_time", 0)) for a in agents]
    if response_times:
        df_response = pd.DataFrame(response_times, columns=["Agent", "Response Time"])
        fig = px.bar(df_response, x="Agent", y="Response Time", 
                    title="Agent Response Times")
        st.plotly_chart(fig, use_container_width=True)
    
    # Success/Error ratio
    st.subheader("Success vs Error Rates")
    
    success_data = []
    for agent in agents:
        name = agent.get("name", "")
        success = agent.get("success_count", 0)
        errors = agent.get("error_count", 0)
        success_data.append({"Agent": name, "Success": success, "Errors": errors})
    
    if success_data:
        df_success = pd.DataFrame(success_data)
        fig = px.bar(df_success, x="Agent", y=["Success", "Errors"], 
                    title="Success vs Error Counts", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    
    # Capability distribution
    st.subheader("Capability Distribution")
    
    all_capabilities = []
    for agent in agents:
        all_capabilities.extend(agent.get("capabilities", []))
    
    if all_capabilities:
        capability_counts = pd.Series(all_capabilities).value_counts()
        fig = px.pie(values=capability_counts.values, names=capability_counts.index,
                    title="Agent Capabilities Distribution")
        st.plotly_chart(fig, use_container_width=True)

def system_admin_page():
    """System administration page"""
    st.header("System Administration")
    
    # System control
    st.subheader("System Control")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart All Agents"):
            st.warning("This feature requires additional implementation")
    
    with col2:
        if st.button("📊 Generate System Report"):
            st.info("Generating comprehensive system report...")
            status = get_system_status()
            st.download_button(
                label="Download Report",
                data=json.dumps(status, indent=2),
                file_name=f"sutazai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🧹 Clear All Tasks"):
            st.warning("This would clear all task history")
    
    # Configuration
    st.subheader("Configuration")
    
    with st.expander("Orchestrator Settings"):
        st.text_input("Health Check Interval (seconds)", value="30")
        st.text_input("Task Timeout (seconds)", value="300")
        st.checkbox("Enable Auto-scaling", value=False)
        st.checkbox("Enable Debug Mode", value=False)
    
    # Logs viewer
    st.subheader("System Logs")
    
    log_type = st.selectbox("Log Type", ["Orchestrator", "All Agents", "Specific Agent"])
    
    if log_type == "Specific Agent":
        agents = get_agents_data()
        agent_name = st.selectbox("Select Agent", [a.get("name", "") for a in agents])
        
        if st.button("View Logs") and agent_name:
            result = make_api_request(f"/api/admin/logs/{agent_name}")
            if "error" not in result:
                st.text_area("Logs", result.get("logs", ""), height=400)
            else:
                st.error(f"Failed to get logs: {result['error']}")

if __name__ == "__main__":
    main()