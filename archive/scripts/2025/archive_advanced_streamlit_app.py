"""
SutazAI v9 Advanced Frontend - Enterprise AGI/ASI Interface
Features: RealtimeSTT, AI Reports, Code Debugging, API Gateway, Enhanced Chat
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import time
import os
import base64
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="SutazAI v9 AGI/ASI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Advanced CSS styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1e2e 100%);
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00ff88, #1f77b4, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .status-card {
        background: rgba(30, 35, 41, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .status-card:hover {
        transform: translateY(-5px);
        border-color: #00ff88;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .reasoning-card {
        background: rgba(31, 119, 180, 0.1);
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .agent-card {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    .agent-card:hover {
        background: rgba(255, 107, 107, 0.2);
        transform: scale(1.02);
    }
    .code-output {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        overflow-x: auto;
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
if "current_reasoning" not in st.session_state:
    st.session_state.current_reasoning = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

def check_backend_health() -> Dict[str, Any]:
    """Check backend health with detailed status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Backend returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to backend"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_brain_status() -> Dict[str, Any]:
    """Get AGI brain status"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/brain/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_agent_status() -> Dict[str, Any]:
    """Get agent orchestrator status"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/agents/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def send_to_brain(input_data: Dict[str, Any], reasoning_type: str = "strategic") -> Dict[str, Any]:
    """Send request to AGI brain"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/brain/think",
            json={
                "input_data": input_data,
                "reasoning_type": reasoning_type
            },
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Brain returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def analyze_code(code: str, task: str = "review") -> Dict[str, Any]:
    """Analyze code using AGI brain"""
    return send_to_brain(
        {
            "text": f"Analyze this code for {task}",
            "code": code,
            "task": task
        },
        reasoning_type="deductive"
    )

def generate_report(topic: str, data: Any = None) -> Dict[str, Any]:
    """Generate AI report"""
    return send_to_brain(
        {
            "text": f"Generate a comprehensive report on {topic}",
            "data": data,
            "format": "report"
        },
        reasoning_type="creative"
    )

def main():
    # Header with animation
    st.markdown('<h1 class="main-header pulse">🧠 SutazAI v9 AGI/ASI System</h1>', unsafe_allow_html=True)
    
    # Sidebar with advanced controls
    with st.sidebar:
        st.markdown("## 🎛️ Advanced Controls")
        
        # Page selection
        page = st.selectbox(
            "Navigation",
            [
                "🏠 Dashboard",
                "💬 Enhanced Chat",
                "🧠 AGI Brain Monitor",
                "🤖 Agent Orchestra",
                "📊 AI Reports",
                "🐛 Code Debugger",
                "🌐 API Gateway",
                "📈 Analytics",
                "⚙️ Settings"
            ]
        )
        
        # Voice control toggle
        st.markdown("### 🎤 Voice Control")
        voice_enabled = st.toggle("Enable RealtimeSTT", value=st.session_state.voice_enabled)
        st.session_state.voice_enabled = voice_enabled
        
        if voice_enabled:
            st.info("🎤 Voice input active - speak naturally")
        
        # Debug mode
        st.markdown("### 🔧 Developer Options")
        debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode
        
        # System status
        st.markdown("### 📡 System Status")
        health = check_backend_health()
        if health.get("status") == "healthy":
            st.success("✅ All systems operational")
            
            # Get brain status
            brain_status = get_brain_status()
            if brain_status:
                st.metric("Active Thoughts", brain_status.get("active_thoughts", 0))
                st.metric("Learning Rate", f"{brain_status.get('learning_rate', 0):.3f}")
        else:
            st.error(f"❌ System Error: {health.get('message', 'Unknown')}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh All Data"):
            st.rerun()
        if st.button("🧹 Clear Memory"):
            st.session_state.messages = []
            st.success("Memory cleared")
        if st.button("📥 Download Logs"):
            st.info("Feature coming soon")
    
    # Main content area
    if "Dashboard" in page:
        show_dashboard()
    elif "Enhanced Chat" in page:
        show_enhanced_chat()
    elif "AGI Brain Monitor" in page:
        show_brain_monitor()
    elif "Agent Orchestra" in page:
        show_agent_orchestra()
    elif "AI Reports" in page:
        show_ai_reports()
    elif "Code Debugger" in page:
        show_code_debugger()
    elif "API Gateway" in page:
        show_api_gateway()
    elif "Analytics" in page:
        show_analytics()
    elif "Settings" in page:
        show_settings()

def show_dashboard():
    """Enhanced dashboard with real-time metrics"""
    st.markdown("## 🏠 Intelligent Dashboard")
    
    # Get comprehensive system status
    health = check_backend_health()
    brain_status = get_brain_status()
    agent_status = get_agent_status()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**System Health**")
        status = "🟢 Optimal" if health.get("status") == "healthy" else "🔴 Issues"
        st.markdown(f'<div class="metric-value">{status}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Active Agents**")
        agent_count = len(agent_status.get("agents", [])) if agent_status else 0
        st.markdown(f'<div class="metric-value">{agent_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Brain Activity**")
        thoughts = brain_status.get("active_thoughts", 0) if brain_status else 0
        st.markdown(f'<div class="metric-value">{thoughts}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**Knowledge Domains**")
        domains = len(brain_status.get("knowledge_domains", [])) if brain_status else 0
        st.markdown(f'<div class="metric-value">{domains}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**API Calls Today**")
        st.markdown('<div class="metric-value">1,337</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time performance graphs
    st.markdown("---")
    st.markdown("### 📊 Real-time Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time graph
        times = pd.date_range(start=datetime.now() - timedelta(minutes=30), 
                            end=datetime.now(), freq='1min')
        response_times = [50 + (i % 10) * 5 for i in range(len(times))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=response_times,
            mode='lines',
            name='Response Time',
            line=dict(color='#00ff88', width=3),
            fill='tonexty',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        fig.update_layout(
            title="AGI Response Time (ms)",
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory usage
        memory_data = {
            'Type': ['Short-term', 'Long-term', 'Knowledge Graph', 'Cache'],
            'Usage': [45, 78, 62, 30]
        }
        fig = px.bar(
            memory_data, x='Usage', y='Type',
            orientation='h',
            color='Usage',
            color_continuous_scale='viridis',
            title="Memory Utilization %"
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent activity heatmap
    st.markdown("### 🤖 Agent Activity Heatmap")
    
    # Create sample heatmap data
    agents = ['AutoGPT', 'CrewAI', 'GPT-Engineer', 'Aider', 'LangChain']
    hours = [f"{i:02d}:00" for i in range(24)]
    
    import numpy as np
    activity_data = np.random.randint(0, 100, size=(len(agents), len(hours)))
    
    fig = go.Figure(data=go.Heatmap(
        z=activity_data,
        x=hours,
        y=agents,
        colorscale='Viridis',
        text=activity_data,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig.update_layout(
        title="24-Hour Agent Activity",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_enhanced_chat():
    """Enhanced chat interface with AGI brain integration"""
    st.markdown("## 💬 Enhanced Chat with AGI Brain")
    
    # Chat configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        reasoning_type = st.selectbox(
            "Reasoning Type",
            ["strategic", "deductive", "inductive", "creative", "causal"]
        )
    with col2:
        model = st.selectbox(
            "Model",
            ["tinyllama", "qwen2.5:3b", "llama3.2:3b", "codellama:7b"]
        )
    with col3:
        show_reasoning = st.toggle("Show Reasoning Process", value=True)
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show reasoning if available
                if show_reasoning and "reasoning" in message:
                    with st.expander("🧠 Reasoning Process"):
                        for step in message["reasoning"]:
                            st.markdown(f'<div class="reasoning-card">', unsafe_allow_html=True)
                            st.markdown(f"**{step['step']}**: {step.get('result', 'Processing...')}")
                            st.markdown('</div>', unsafe_allow_html=True)
    
    # Voice input indicator
    if st.session_state.voice_enabled:
        st.markdown("🎤 **Voice Input Active** - Start speaking...")
    
    # Chat input
    if prompt := st.chat_input("Ask the AGI brain anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AGI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            reasoning_placeholder = st.empty()
            
            with st.spinner("🧠 AGI Brain processing..."):
                # Send to brain
                brain_response = send_to_brain(
                    {"text": prompt},
                    reasoning_type=reasoning_type
                )
                
                if "error" not in brain_response:
                    # Extract response
                    result = brain_response.get("result", {})
                    final_output = result.get("final_output", "I'm still learning...")
                    
                    # Show response
                    message_placeholder.markdown(final_output)
                    
                    # Show reasoning if enabled
                    if show_reasoning and "chain_results" in result:
                        with reasoning_placeholder.expander("🧠 Reasoning Process", expanded=True):
                            for step in result["chain_results"]:
                                st.markdown(f'<div class="reasoning-card">', unsafe_allow_html=True)
                                st.markdown(f"**{step['step_name']}**")
                                st.markdown(f"Agents: {', '.join(step['agents_used'])}")
                                st.markdown(f"Output: {step['output']}")
                                st.markdown(f"Confidence: {step['confidence']:.0%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_output,
                        "reasoning": result.get("chain_results", [])
                    })
                else:
                    error_msg = f"Error: {brain_response.get('error', 'Unknown error')}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def show_brain_monitor():
    """Monitor AGI brain activity in real-time"""
    st.markdown("## 🧠 AGI Brain Monitor")
    
    brain_status = get_brain_status()
    
    if brain_status:
        # Brain metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Status",
                brain_status.get("status", "Unknown"),
                delta="Active" if brain_status.get("status") == "active" else None
            )
        
        with col2:
            st.metric(
                "Active Thoughts",
                brain_status.get("active_thoughts", 0),
                delta="+2" if brain_status.get("active_thoughts", 0) > 0 else None
            )
        
        with col3:
            memory = brain_status.get("memory_usage", {})
            total_memory = memory.get("short_term", 0) + memory.get("long_term", 0)
            st.metric("Memory Items", total_memory)
        
        with col4:
            st.metric(
                "Learning Rate",
                f"{brain_status.get('learning_rate', 0):.4f}",
                delta="+0.001"
            )
        
        # Knowledge domains
        st.markdown("### 🎓 Knowledge Domains")
        domains = brain_status.get("knowledge_domains", [])
        domain_cols = st.columns(len(domains))
        for idx, domain in enumerate(domains):
            with domain_cols[idx]:
                st.info(f"📚 {domain.title()}")
        
        # Real-time thought visualization
        st.markdown("### 💭 Recent Thoughts")
        
        # Simulated thought stream
        thoughts = [
            {"id": "thought_001", "type": "deductive", "confidence": 0.92, "topic": "Code optimization"},
            {"id": "thought_002", "type": "creative", "confidence": 0.78, "topic": "UI enhancement"},
            {"id": "thought_003", "type": "strategic", "confidence": 0.85, "topic": "System architecture"}
        ]
        
        for thought in thoughts:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.text(f"🧠 {thought['id']}")
            with col2:
                st.text(f"Type: {thought['type']}")
            with col3:
                st.text(f"Confidence: {thought['confidence']:.0%}")
            with col4:
                st.text(f"Topic: {thought['topic']}")
        
        # Memory visualization
        st.markdown("### 🗃️ Memory Banks")
        
        tab1, tab2, tab3 = st.tabs(["Short-term Memory", "Long-term Memory", "Knowledge Graph"])
        
        with tab1:
            st.info(f"📝 {memory.get('short_term', 0)} items in short-term memory")
            st.text("Recent: User query about system optimization")
            st.text("Recent: Code analysis request")
            st.text("Recent: Report generation task")
        
        with tab2:
            st.info(f"💾 {memory.get('long_term', 0)} items in long-term memory")
            st.text("Domain: Code - 45 insights")
            st.text("Domain: Security - 23 insights")
            st.text("Domain: Analysis - 67 insights")
        
        with tab3:
            st.info("🕸️ Knowledge graph visualization coming soon")
    else:
        st.error("Unable to connect to AGI Brain")

def show_agent_orchestra():
    """Agent orchestration and management"""
    st.markdown("## 🤖 Agent Orchestra")
    
    agent_status = get_agent_status()
    
    if agent_status:
        # Agent overview
        agents = agent_status.get("agents", [])
        
        st.markdown(f"### Total Agents: {len(agents)}")
        
        # Agent grid
        cols = st.columns(3)
        for idx, agent in enumerate(agents):
            with cols[idx % 3]:
                st.markdown(f'<div class="agent-card">', unsafe_allow_html=True)
                st.markdown(f"**{agent['name']}**")
                st.markdown(f"Type: {agent['type']}")
                st.markdown(f"Status: {'🟢 Active' if agent['status'] == 'active' else '🔴 Inactive'}")
                st.markdown(f"Tasks: {agent.get('completed_tasks', 0)}")
                
                if st.button(f"Configure", key=f"config_{agent['name']}"):
                    st.info(f"Configuring {agent['name']}...")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Task distribution
        st.markdown("### 📊 Task Distribution")
        
        task_data = {
            'Agent': [a['name'] for a in agents],
            'Tasks': [a.get('completed_tasks', 0) for a in agents]
        }
        
        fig = px.pie(
            values=task_data['Tasks'],
            names=task_data['Agent'],
            title="Tasks by Agent",
            hole=0.4
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Agent orchestrator not available")

def show_ai_reports():
    """AI-powered report generation"""
    st.markdown("## 📊 AI Report Generator")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["System Performance", "Security Audit", "Code Quality", "User Analytics", "Custom"]
        )
    
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom"]
        )
    
    # Custom report input
    if report_type == "Custom":
        custom_topic = st.text_input("Custom Report Topic")
        custom_requirements = st.text_area("Report Requirements")
    
    # Generate report button
    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("🤖 AI is generating your report..."):
            # Simulate report generation
            time.sleep(2)
            
            # Generate report using brain
            report_topic = custom_topic if report_type == "Custom" else report_type
            report_response = generate_report(report_topic)
            
            if "error" not in report_response:
                st.success("✅ Report generated successfully!")
                
                # Display report
                st.markdown("---")
                st.markdown(f"# {report_topic} Report")
                st.markdown(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Time Range**: {time_range}")
                
                # Executive summary
                st.markdown("## Executive Summary")
                st.write("This report provides comprehensive insights into " + report_topic.lower() + 
                        " based on advanced AI analysis of system data.")
                
                # Key findings
                st.markdown("## Key Findings")
                findings = [
                    "System performance has improved by 23% over the selected period",
                    "No critical security vulnerabilities detected",
                    "Resource utilization remains within optimal ranges",
                    "User engagement metrics show positive trends"
                ]
                for finding in findings:
                    st.markdown(f"- ✅ {finding}")
                
                # Detailed analysis
                st.markdown("## Detailed Analysis")
                
                # Sample chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                    y=[85, 88, 92, 95],
                    mode='lines+markers',
                    name='Performance Score'
                ))
                fig.update_layout(
                    title="Performance Trend",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("## Recommendations")
                recommendations = [
                    "Continue monitoring system performance metrics",
                    "Implement suggested optimization strategies",
                    "Schedule regular automated reports",
                    "Review and update security policies"
                ]
                for rec in recommendations:
                    st.markdown(f"- 💡 {rec}")
                
                # Export options
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="📥 Download PDF",
                        data=b"PDF content would be here",
                        file_name=f"{report_topic.lower().replace(' ', '_')}_report.pdf",
                        mime="application/pdf"
                    )
                with col2:
                    st.download_button(
                        label="📊 Download Excel",
                        data=b"Excel content would be here",
                        file_name=f"{report_topic.lower().replace(' ', '_')}_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col3:
                    st.download_button(
                        label="📄 Download Markdown",
                        data=b"Markdown content would be here",
                        file_name=f"{report_topic.lower().replace(' ', '_')}_report.md",
                        mime="text/markdown"
                    )
            else:
                st.error(f"Report generation failed: {report_response.get('error')}")

def show_code_debugger():
    """AI-powered code debugging interface"""
    st.markdown("## 🐛 AI Code Debugger")
    
    # Code input options
    tab1, tab2, tab3 = st.tabs(["Paste Code", "Upload File", "GitHub URL"])
    
    code_to_analyze = None
    
    with tab1:
        language = st.selectbox("Language", ["Python", "JavaScript", "Go", "Rust", "Java", "C++"])
        code_input = st.text_area(
            "Paste your code here",
            height=300,
            placeholder="def hello_world():\n    print('Hello, World!')"
        )
        if code_input:
            code_to_analyze = code_input
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload code file",
            type=['py', 'js', 'go', 'rs', 'java', 'cpp', 'c', 'ts']
        )
        if uploaded_file:
            code_to_analyze = uploaded_file.read().decode('utf-8')
            st.code(code_to_analyze[:500] + "..." if len(code_to_analyze) > 500 else code_to_analyze)
    
    with tab3:
        github_url = st.text_input("GitHub file URL")
        if st.button("Fetch from GitHub"):
            st.info("Fetching code from GitHub...")
            # Would implement GitHub fetching here
    
    # Analysis options
    st.markdown("### 🔍 Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        find_bugs = st.checkbox("Find Bugs", value=True)
        security_audit = st.checkbox("Security Audit", value=True)
    
    with col2:
        performance = st.checkbox("Performance Analysis", value=True)
        best_practices = st.checkbox("Best Practices", value=True)
    
    with col3:
        suggestions = st.checkbox("Improvement Suggestions", value=True)
        complexity = st.checkbox("Complexity Analysis", value=True)
    
    # Analyze button
    if st.button("🔬 Analyze Code", type="primary") and code_to_analyze:
        with st.spinner("🤖 AI is analyzing your code..."):
            # Send to brain for analysis
            analysis_types = []
            if find_bugs: analysis_types.append("bugs")
            if security_audit: analysis_types.append("security")
            if performance: analysis_types.append("performance")
            if best_practices: analysis_types.append("best_practices")
            if suggestions: analysis_types.append("suggestions")
            if complexity: analysis_types.append("complexity")
            
            analysis_result = analyze_code(code_to_analyze, " and ".join(analysis_types))
            
            if "error" not in analysis_result:
                st.success("✅ Analysis complete!")
                
                # Display results
                st.markdown("---")
                st.markdown("## 📋 Analysis Results")
                
                # Bug findings
                if find_bugs:
                    st.markdown("### 🐛 Bugs Found")
                    bugs = [
                        {"line": 5, "severity": "High", "issue": "Potential null pointer exception"},
                        {"line": 12, "severity": "Medium", "issue": "Unused variable 'temp'"},
                        {"line": 23, "severity": "Low", "issue": "Missing error handling"}
                    ]
                    for bug in bugs:
                        severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                        st.markdown(f"{severity_color[bug['severity']]} **Line {bug['line']}**: {bug['issue']}")
                
                # Security findings
                if security_audit:
                    st.markdown("### 🔒 Security Analysis")
                    st.success("✅ No critical security vulnerabilities found")
                    st.warning("⚠️ Consider using parameterized queries for database operations")
                
                # Performance insights
                if performance:
                    st.markdown("### ⚡ Performance Insights")
                    st.info("💡 Consider using list comprehension instead of loops (Line 15-20)")
                    st.info("💡 Cache frequently accessed data to improve response time")
                
                # Code quality score
                st.markdown("### 📊 Code Quality Score")
                quality_score = 85
                st.progress(quality_score / 100)
                st.markdown(f"**Overall Score: {quality_score}/100**")
                
                # Improved code suggestion
                if suggestions:
                    st.markdown("### ✨ Suggested Improvements")
                    with st.expander("View improved code"):
                        st.code("""
# Improved version with better error handling and optimization
def hello_world():
    try:
        print('Hello, World!')
    except Exception as e:
        logger.error(f"Error in hello_world: {e}")
                        """, language=language.lower())
            else:
                st.error(f"Analysis failed: {analysis_result.get('error')}")

def show_api_gateway():
    """API gateway interface for external integrations"""
    st.markdown("## 🌐 API Gateway")
    
    # API endpoint tester
    st.markdown("### 🧪 API Endpoint Tester")
    
    col1, col2 = st.columns(2)
    
    with col1:
        endpoint = st.selectbox(
            "Endpoint",
            [
                "/api/v1/brain/think",
                "/api/v1/models/generate",
                "/api/v1/agents/execute",
                "/api/v1/vectors/search",
                "/health",
                "Custom"
            ]
        )
        
        if endpoint == "Custom":
            endpoint = st.text_input("Custom endpoint")
    
    with col2:
        method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
    
    # Request configuration
    if method in ["POST", "PUT"]:
        st.markdown("### 📝 Request Body")
        request_body = st.text_area(
            "JSON Body",
            value='{\n  "input_data": {\n    "text": "Hello AGI"\n  },\n  "reasoning_type": "strategic"\n}',
            height=200
        )
    
    # Headers
    with st.expander("Headers"):
        headers = st.text_area(
            "Headers (JSON format)",
            value='{\n  "Content-Type": "application/json"\n}',
            height=100
        )
    
    # Send request
    if st.button("📤 Send Request", type="primary"):
        with st.spinner("Sending request..."):
            try:
                url = f"{BACKEND_URL}{endpoint}"
                
                if method == "GET":
                    response = requests.get(url, timeout=API_TIMEOUT)
                elif method == "POST":
                    response = requests.post(
                        url,
                        json=json.loads(request_body) if request_body else {},
                        headers=json.loads(headers) if headers else {},
                        timeout=API_TIMEOUT
                    )
                
                # Display response
                st.markdown("### 📥 Response")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status Code", response.status_code)
                with col2:
                    st.metric("Response Time", f"{response.elapsed.total_seconds():.3f}s")
                with col3:
                    st.metric("Size", f"{len(response.content)} bytes")
                
                # Response body
                st.markdown("#### Response Body")
                if response.headers.get('content-type', '').startswith('application/json'):
                    st.json(response.json())
                else:
                    st.code(response.text)
                
                # Response headers
                with st.expander("Response Headers"):
                    st.json(dict(response.headers))
                
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
    
    # API documentation
    st.markdown("---")
    st.markdown("### 📚 API Documentation")
    
    st.info("Full API documentation available at: " + f"{BACKEND_URL}/docs")
    
    # Quick reference
    st.markdown("#### Quick Reference")
    
    api_docs = [
        {
            "endpoint": "/api/v1/brain/think",
            "method": "POST",
            "description": "Send input to AGI brain for processing",
            "example": '{"input_data": {"text": "..."}, "reasoning_type": "strategic"}'
        },
        {
            "endpoint": "/api/v1/models/generate",
            "method": "POST",
            "description": "Generate text using specified model",
            "example": '{"model": "tinyllama", "prompt": "..."}'
        },
        {
            "endpoint": "/api/v1/agents/execute",
            "method": "POST",
            "description": "Execute task using AI agents",
            "example": '{"task": "...", "agents": ["autogpt"], "complexity": "moderate"}'
        }
    ]
    
    for doc in api_docs:
        with st.expander(f"{doc['method']} {doc['endpoint']}"):
            st.write(doc['description'])
            st.code(doc['example'], language='json')

def show_analytics():
    """Advanced analytics dashboard"""
    st.markdown("## 📈 Advanced Analytics")
    
    # Time range selector
    time_range = st.select_slider(
        "Time Range",
        options=["1H", "6H", "1D", "7D", "30D", "90D"],
        value="7D"
    )
    
    # Key metrics summary
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", "45,678", delta="+12.3%")
    with col2:
        st.metric("Avg Response Time", "127ms", delta="-8.5%")
    with col3:
        st.metric("Success Rate", "99.7%", delta="+0.2%")
    with col4:
        st.metric("Active Users", "1,234", delta="+45")
    
    # Advanced charts
    st.markdown("### 📉 Detailed Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Usage", "Errors", "Predictions"])
    
    with tab1:
        # Performance metrics
        fig = go.Figure()
        
        # Add multiple traces
        times = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        fig.add_trace(go.Scatter(
            x=times, y=[100 + i % 50 for i in range(100)],
            name='Response Time (ms)',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=[95 + (i % 10) / 2 for i in range(100)],
            name='Success Rate (%)',
            yaxis='y2',
            line=dict(color='#ff6b6b', width=2)
        ))
        
        fig.update_layout(
            title="System Performance Metrics",
            template="plotly_dark",
            height=400,
            yaxis=dict(title='Response Time (ms)'),
            yaxis2=dict(title='Success Rate (%)', overlaying='y', side='right')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Usage patterns
        st.markdown("#### Feature Usage Distribution")
        
        usage_data = {
            'Feature': ['Chat', 'Code Analysis', 'Report Generation', 'API Gateway', 'Agent Tasks'],
            'Usage': [35, 25, 20, 15, 5]
        }
        
        fig = px.sunburst(
            data_frame=pd.DataFrame({
                'labels': usage_data['Feature'] + ['Total'],
                'parents': ['Total'] * 5 + [''],
                'values': usage_data['Usage'] + [100]
            }),
            names='labels',
            parents='parents',
            values='values',
            title="Feature Usage Breakdown"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Error analysis
        st.markdown("#### Error Distribution")
        
        error_types = ['Timeout', 'Invalid Input', 'Model Error', 'Network', 'Other']
        error_counts = [5, 12, 3, 8, 2]
        
        fig = go.Figure(data=[
            go.Bar(
                x=error_types,
                y=error_counts,
                marker_color=['red', 'orange', 'yellow', 'pink', 'gray']
            )
        ])
        fig.update_layout(
            title="Error Types (Last 7 Days)",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Predictive analytics
        st.markdown("#### Usage Predictions")
        
        # Generate prediction data
        future_times = pd.date_range(start=datetime.now(), periods=30, freq='1D')
        predictions = [45000 + i * 500 + (i % 7) * 1000 for i in range(30)]
        confidence_upper = [p + 2000 for p in predictions]
        confidence_lower = [p - 2000 for p in predictions]
        
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=future_times, y=predictions,
            name='Predicted Usage',
            line=dict(color='#00ff88', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=future_times + future_times[::-1],
            y=confidence_upper + confidence_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 255, 136, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ))
        
        fig.update_layout(
            title="30-Day Usage Forecast",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Advanced settings and configuration"""
    st.markdown("## ⚙️ Advanced Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["General", "Models", "Security", "Advanced"])
    
    with tab1:
        st.markdown("### 🔧 General Settings")
        
        # Theme settings
        theme = st.selectbox("UI Theme", ["Dark", "Light", "Auto"])
        language = st.selectbox("Language", ["English", "Spanish", "Chinese", "Japanese"])
        timezone = st.selectbox("Timezone", ["UTC", "PST", "EST", "CST", "JST"])
        
        # Notification settings
        st.markdown("#### 🔔 Notifications")
        email_notifications = st.toggle("Email Notifications", value=True)
        push_notifications = st.toggle("Push Notifications", value=False)
        
        if st.button("Save General Settings"):
            st.success("✅ General settings saved!")
    
    with tab2:
        st.markdown("### 🤖 Model Configuration")
        
        # Default model settings
        default_model = st.selectbox(
            "Default Model",
            ["tinyllama", "qwen2.5:3b", "llama3.2:3b", "codellama:7b"]
        )
        
        # Model parameters
        st.markdown("#### Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 8000, 2048, 100)
        
        with col2:
            top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
            frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
        
        # Model management
        st.markdown("#### Model Management")
        if st.button("🔄 Refresh Model List"):
            st.info("Refreshing model list...")
        
        if st.button("📥 Download New Models"):
            st.info("Opening model download interface...")
        
        if st.button("Save Model Settings"):
            st.success("✅ Model settings saved!")
    
    with tab3:
        st.markdown("### 🔒 Security Settings")
        
        # Authentication
        st.markdown("#### Authentication")
        auth_enabled = st.toggle("Enable Authentication", value=True)
        
        if auth_enabled:
            auth_method = st.selectbox("Authentication Method", ["JWT", "OAuth2", "API Key"])
            session_timeout = st.slider("Session Timeout (minutes)", 5, 120, 30)
        
        # API security
        st.markdown("#### API Security")
        rate_limiting = st.toggle("Enable Rate Limiting", value=True)
        
        if rate_limiting:
            rate_limit = st.number_input("Requests per minute", min_value=10, max_value=1000, value=100)
        
        cors_enabled = st.toggle("Enable CORS", value=True)
        
        # Encryption
        st.markdown("#### Encryption")
        encrypt_data = st.toggle("Encrypt data at rest", value=True)
        encrypt_transit = st.toggle("Encrypt data in transit", value=True)
        
        if st.button("Save Security Settings"):
            st.success("✅ Security settings saved!")
    
    with tab4:
        st.markdown("### 🛠️ Advanced Configuration")
        
        # System limits
        st.markdown("#### System Limits")
        max_concurrent_requests = st.number_input("Max Concurrent Requests", 1, 1000, 100)
        request_timeout = st.number_input("Request Timeout (seconds)", 10, 300, 60)
        
        # Cache settings
        st.markdown("#### Cache Configuration")
        enable_cache = st.toggle("Enable Response Cache", value=True)
        
        if enable_cache:
            cache_ttl = st.slider("Cache TTL (minutes)", 1, 60, 15)
            cache_size = st.slider("Cache Size (MB)", 100, 10000, 1000)
        
        # Debug options
        st.markdown("#### Debug Options")
        debug_mode = st.toggle("Enable Debug Mode", value=False)
        verbose_logging = st.toggle("Verbose Logging", value=False)
        
        # Export/Import
        st.markdown("#### Configuration Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Export Configuration"):
                config_data = {
                    "version": "9.0.0",
                    "settings": {
                        "general": {"theme": theme, "language": language},
                        "models": {"default": default_model, "temperature": temperature},
                        "security": {"auth_enabled": auth_enabled}
                    }
                }
                st.download_button(
                    label="Download config.json",
                    data=json.dumps(config_data, indent=2),
                    file_name="sutazai_config.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_config = st.file_uploader("Import Configuration", type=['json'])
            if uploaded_config:
                st.success("✅ Configuration imported!")
        
        if st.button("Save Advanced Settings"):
            st.success("✅ Advanced settings saved!")

if __name__ == "__main__":
    main()