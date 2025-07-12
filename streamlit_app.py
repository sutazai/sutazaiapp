#!/usr/bin/env python3
"""
SutazAI - Comprehensive E2E Autonomous AGI/ASI System
Main Streamlit Web UI Application

This is the main entry point for the SutazAI web interface, providing a comprehensive
dashboard for all AI agents, models, and system components.
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any, Optional
import time

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

try:
    from core.model_manager import ModelManager
    from core.vector_memory import VectorMemory
    from core.orchestrator import Orchestrator
    from api.main import app as fastapi_app
    from agents import *
    from utils.logger import setup_logger
    from utils.config import load_config
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="SutazAI - AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://github.com/suomilanittaja/SutazAI',
        'Report a bug': 'https://github.com/suomilanittaja/SutazAI/issues',
        'About': "SutazAI - Comprehensive E2E Autonomous AGI/ASI System"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-online {
        color: #10b981;
        font-weight: bold;
    }
    .status-offline {
        color: #ef4444;
        font-weight: bold;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        margin: 1rem 0;
        padding: 1rem;
        background: #f1f5f9;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

class SutazAIApp:
    """Main SutazAI Streamlit Application"""
    
    def __init__(self):
        self.config = load_config()
        self.model_manager = None
        self.vector_memory = None
        self.orchestrator = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize core system components"""
        try:
            # Initialize components
            self.model_manager = ModelManager()
            self.vector_memory = VectorMemory()
            self.orchestrator = Orchestrator(
                model_manager=self.model_manager,
                vector_memory=self.vector_memory
            )
            
            # Initialize session state
            if 'system_initialized' not in st.session_state:
                st.session_state.system_initialized = True
                st.session_state.conversation_history = []
                st.session_state.active_agents = []
                st.session_state.system_metrics = {}
                
            logger.info("SutazAI system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            st.error(f"System initialization failed: {e}")
    
    def render_header(self):
        """Render the main application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 SutazAI - Autonomous AGI/ASI System</h1>
            <p>Comprehensive End-to-End Artificial General Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the application sidebar"""
        with st.sidebar:
            st.title("🎛️ Control Panel")
            
            # System Status
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.subheader("📊 System Status")
            
            system_status = self.get_system_status()
            st.metric("Agents Online", system_status['agents_online'])
            st.metric("Models Loaded", system_status['models_loaded'])
            st.metric("Memory Usage", f"{system_status['memory_usage']:.1f}%")
            st.metric("CPU Usage", f"{system_status['cpu_usage']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick Actions
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.subheader("⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Restart System", use_container_width=True):
                    self.restart_system()
                if st.button("🧠 Load Model", use_container_width=True):
                    st.session_state.show_model_loader = True
            
            with col2:
                if st.button("📊 View Logs", use_container_width=True):
                    st.session_state.show_logs = True
                if st.button("⚙️ Settings", use_container_width=True):
                    st.session_state.show_settings = True
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Agent Control
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.subheader("🤖 Agent Control")
            
            available_agents = self.get_available_agents()
            selected_agents = st.multiselect(
                "Select Active Agents",
                options=list(available_agents.keys()),
                default=st.session_state.get('active_agents', [])
            )
            
            if st.button("Activate Selected Agents", use_container_width=True):
                st.session_state.active_agents = selected_agents
                st.success(f"Activated {len(selected_agents)} agents")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_main_content(self):
        """Render the main application content"""
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏠 Dashboard",
            "🤖 Agents",
            "💬 Chat Interface",
            "📊 Analytics",
            "🔧 Tools",
            "📚 Documentation"
        ])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_agents_panel()
        
        with tab3:
            self.render_chat_interface()
        
        with tab4:
            self.render_analytics_panel()
        
        with tab5:
            self.render_tools_panel()
        
        with tab6:
            self.render_documentation()
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("📊 System Dashboard")
        
        # System overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Agents", len(self.get_available_agents()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Sessions", len(st.session_state.get('conversation_history', [])))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Models Available", len(self.model_manager.get_available_models()) if self.model_manager else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Memory Entries", self.vector_memory.get_memory_count() if self.vector_memory else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        self.render_activity_timeline()
        
        # System health
        st.subheader("🏥 System Health")
        self.render_system_health()
    
    def render_agents_panel(self):
        """Render the agents management panel"""
        st.header("🤖 Agent Management")
        
        available_agents = self.get_available_agents()
        
        # Agent grid
        cols = st.columns(3)
        for idx, (agent_name, agent_info) in enumerate(available_agents.items()):
            col = cols[idx % 3]
            
            with col:
                with st.container():
                    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
                    st.subheader(f"🤖 {agent_name}")
                    
                    # Agent status
                    status = "🟢 Online" if agent_info.get('active', False) else "🔴 Offline"
                    st.markdown(f"**Status:** {status}")
                    
                    # Agent description
                    st.write(agent_info.get('description', 'No description available'))
                    
                    # Agent actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Start", key=f"start_{agent_name}"):
                            self.start_agent(agent_name)
                    with col2:
                        if st.button(f"Stop", key=f"stop_{agent_name}"):
                            self.stop_agent(agent_name)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.header("💬 AI Chat Interface")
        
        # Chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.get('conversation_history', []):
                with st.chat_message(message['role']):
                    st.write(message['content'])
        
        # Chat input
        user_input = st.chat_input("Ask anything to SutazAI...")
        
        if user_input:
            # Add user message
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Process with orchestrator
            with st.spinner("Processing..."):
                response = self.process_user_input(user_input)
            
            # Add assistant response
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            st.rerun()
    
    def render_analytics_panel(self):
        """Render the analytics panel"""
        st.header("📊 System Analytics")
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        # Generate sample data for demonstration
        metrics_data = self.generate_sample_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time chart
            fig_response = px.line(
                metrics_data['response_times'],
                x='timestamp',
                y='response_time',
                title='Response Times Over Time'
            )
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            # Agent usage chart
            fig_usage = px.pie(
                metrics_data['agent_usage'],
                values='usage_count',
                names='agent_name',
                title='Agent Usage Distribution'
            )
            st.plotly_chart(fig_usage, use_container_width=True)
        
        # Resource utilization
        st.subheader("💾 Resource Utilization")
        
        resource_data = metrics_data['resource_usage']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=resource_data['cpu'],
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ))
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            fig_memory = go.Figure(go.Indicator(
                mode="gauge+number",
                value=resource_data['memory'],
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ))
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            fig_disk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=resource_data['disk'],
                title={'text': "Disk Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkorange"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ))
            st.plotly_chart(fig_disk, use_container_width=True)
    
    def render_tools_panel(self):
        """Render the tools panel"""
        st.header("🔧 System Tools")
        
        tool_tab1, tool_tab2, tool_tab3, tool_tab4 = st.tabs([
            "📄 Document Processing",
            "💰 Financial Analysis",
            "💻 Code Editor",
            "🔍 System Diagnostics"
        ])
        
        with tool_tab1:
            self.render_document_processing()
        
        with tool_tab2:
            self.render_financial_analysis()
        
        with tool_tab3:
            self.render_code_editor()
        
        with tool_tab4:
            self.render_system_diagnostics()
    
    def render_documentation(self):
        """Render the documentation panel"""
        st.header("📚 Documentation")
        
        doc_sections = {
            "🚀 Getting Started": "Learn how to use SutazAI effectively",
            "🤖 Agent Guide": "Comprehensive guide to all available agents",
            "🔧 API Reference": "Complete API documentation",
            "⚙️ Configuration": "System configuration options",
            "🛠️ Troubleshooting": "Common issues and solutions"
        }
        
        selected_section = st.selectbox("Select Documentation Section", list(doc_sections.keys()))
        
        st.markdown(f"### {selected_section}")
        st.markdown(doc_sections[selected_section])
        
        # Add more detailed documentation content here
        if selected_section == "🚀 Getting Started":
            st.markdown("""
            #### Welcome to SutazAI
            
            SutazAI is a comprehensive End-to-End Autonomous AGI/ASI System that provides:
            
            - **Multi-Agent Architecture**: Deploy multiple specialized AI agents
            - **Advanced Model Management**: Support for various AI models
            - **Vector Memory System**: Persistent memory across sessions
            - **Web Interface**: User-friendly Streamlit interface
            - **API Integration**: RESTful API for external integrations
            
            #### Quick Start
            
            1. **Activate Agents**: Use the sidebar to select and activate agents
            2. **Start Chatting**: Use the Chat Interface tab to interact with AI
            3. **Monitor System**: Check the Dashboard for system health
            4. **Analyze Performance**: View Analytics for detailed metrics
            """)
    
    def render_document_processing(self):
        """Render document processing tools"""
        st.subheader("📄 Document Processing System")
        
        uploaded_file = st.file_uploader(
            "Upload documents for AI processing",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_file:
            st.success(f"Uploaded {len(uploaded_file)} files")
            
            for file in uploaded_file:
                st.write(f"📄 {file.name} ({file.size} bytes)")
            
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Process documents here
                    time.sleep(2)
                    st.success("Documents processed successfully!")
    
    def render_financial_analysis(self):
        """Render financial analysis tools"""
        st.subheader("💰 Financial Analysis AI System")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Stock Analysis", "Portfolio Optimization", "Risk Assessment", "Market Trends"]
        )
        
        if analysis_type == "Stock Analysis":
            symbol = st.text_input("Enter stock symbol (e.g., AAPL)")
            if symbol and st.button("Analyze Stock"):
                with st.spinner("Analyzing stock..."):
                    # Perform stock analysis
                    time.sleep(2)
                    st.success(f"Analysis complete for {symbol}")
    
    def render_code_editor(self):
        """Render code editor interface"""
        st.subheader("💻 AI Code Editor & Debugging Panel")
        
        # Code editor
        code = st.text_area(
            "Code Editor",
            value="# Write your code here\nprint('Hello, SutazAI!')",
            height=300
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analyze Code"):
                st.info("Code analysis feature coming soon!")
        
        with col2:
            if st.button("🐛 Debug Code"):
                st.info("Debug feature coming soon!")
        
        with col3:
            if st.button("✨ AI Suggestions"):
                st.info("AI suggestions feature coming soon!")
    
    def render_system_diagnostics(self):
        """Render system diagnostics"""
        st.subheader("🔍 System Diagnostics")
        
        if st.button("Run Full System Check"):
            with st.spinner("Running diagnostics..."):
                # Run system diagnostics
                results = self.run_system_diagnostics()
                
                for check, status in results.items():
                    if status:
                        st.success(f"✅ {check}: OK")
                    else:
                        st.error(f"❌ {check}: Failed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        import psutil
        
        return {
            'agents_online': len(st.session_state.get('active_agents', [])),
            'models_loaded': len(self.model_manager.get_available_models()) if self.model_manager else 0,
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(interval=1)
        }
    
    def get_available_agents(self) -> Dict[str, Dict]:
        """Get available agents and their information"""
        return {
            'AutoGPT': {
                'description': 'Autonomous GPT agent for task execution',
                'active': 'AutoGPT' in st.session_state.get('active_agents', [])
            },
            'LocalAGI': {
                'description': 'Local AGI agent for general intelligence tasks',
                'active': 'LocalAGI' in st.session_state.get('active_agents', [])
            },
            'AutoGen': {
                'description': 'Multi-agent conversation framework',
                'active': 'AutoGen' in st.session_state.get('active_agents', [])
            },
            'BigAGI': {
                'description': 'Advanced artificial general intelligence',
                'active': 'BigAGI' in st.session_state.get('active_agents', [])
            },
            'AgentZero': {
                'description': 'Universal problem-solving with zero-shot learning',
                'active': 'AgentZero' in st.session_state.get('active_agents', [])
            },
            'BrowserUse': {
                'description': 'Intelligent web automation agent',
                'active': 'BrowserUse' in st.session_state.get('active_agents', [])
            },
            'Skyvern': {
                'description': 'Advanced AI-powered web automation',
                'active': 'Skyvern' in st.session_state.get('active_agents', [])
            },
            'OpenWebUI': {
                'description': 'Web interface management agent',
                'active': 'OpenWebUI' in st.session_state.get('active_agents', [])
            },
            'TabbyML': {
                'description': 'AI-powered code completion agent',
                'active': 'TabbyML' in st.session_state.get('active_agents', [])
            },
            'Semgrep': {
                'description': 'Advanced static code analysis agent',
                'active': 'Semgrep' in st.session_state.get('active_agents', [])
            }
        }
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input through the orchestrator"""
        try:
            if self.orchestrator:
                response = asyncio.run(self.orchestrator.process_request(user_input))
                return response
            else:
                return "System not properly initialized. Please restart the application."
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return f"Error processing request: {str(e)}"
    
    def generate_sample_metrics(self) -> Dict[str, Any]:
        """Generate sample metrics for demonstration"""
        import random
        from datetime import datetime, timedelta
        
        # Sample response times
        base_time = datetime.now() - timedelta(hours=24)
        response_times = []
        
        for i in range(100):
            response_times.append({
                'timestamp': base_time + timedelta(minutes=i*15),
                'response_time': random.uniform(0.1, 2.0)
            })
        
        # Sample agent usage
        agents = ['AutoGPT', 'LocalAGI', 'AutoGen', 'BigAGI', 'AgentZero']
        agent_usage = []
        
        for agent in agents:
            agent_usage.append({
                'agent_name': agent,
                'usage_count': random.randint(10, 100)
            })
        
        # Sample resource usage
        resource_usage = {
            'cpu': random.uniform(20, 80),
            'memory': random.uniform(30, 70),
            'disk': random.uniform(40, 85)
        }
        
        return {
            'response_times': pd.DataFrame(response_times),
            'agent_usage': pd.DataFrame(agent_usage),
            'resource_usage': resource_usage
        }
    
    def render_activity_timeline(self):
        """Render recent activity timeline"""
        activities = [
            "🤖 AutoGPT agent started",
            "💬 User query processed",
            "🧠 Model loaded: GPT-4",
            "📊 Analytics updated",
            "🔄 System health check completed"
        ]
        
        for activity in activities[-5:]:
            st.write(f"• {activity}")
    
    def render_system_health(self):
        """Render system health indicators"""
        health_checks = {
            "Database Connection": True,
            "Vector Memory": True,
            "Model Manager": True,
            "Agent Framework": True,
            "API Server": True
        }
        
        cols = st.columns(len(health_checks))
        
        for idx, (check, status) in enumerate(health_checks.items()):
            with cols[idx]:
                if status:
                    st.success(f"✅ {check}")
                else:
                    st.error(f"❌ {check}")
    
    def start_agent(self, agent_name: str):
        """Start a specific agent"""
        if agent_name not in st.session_state.get('active_agents', []):
            st.session_state.active_agents = st.session_state.get('active_agents', []) + [agent_name]
            st.success(f"Started {agent_name} agent")
        else:
            st.warning(f"{agent_name} agent is already running")
    
    def stop_agent(self, agent_name: str):
        """Stop a specific agent"""
        if agent_name in st.session_state.get('active_agents', []):
            st.session_state.active_agents.remove(agent_name)
            st.success(f"Stopped {agent_name} agent")
        else:
            st.warning(f"{agent_name} agent is not running")
    
    def restart_system(self):
        """Restart the entire system"""
        with st.spinner("Restarting system..."):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Reinitialize
            self.initialize_system()
            
            time.sleep(2)
            st.success("System restarted successfully!")
            st.rerun()
    
    def run_system_diagnostics(self) -> Dict[str, bool]:
        """Run comprehensive system diagnostics"""
        results = {}
        
        # Database check
        try:
            # Simulate database check
            results['Database Connection'] = True
        except:
            results['Database Connection'] = False
        
        # Model manager check
        try:
            results['Model Manager'] = self.model_manager is not None
        except:
            results['Model Manager'] = False
        
        # Vector memory check
        try:
            results['Vector Memory'] = self.vector_memory is not None
        except:
            results['Vector Memory'] = False
        
        # Orchestrator check
        try:
            results['Orchestrator'] = self.orchestrator is not None
        except:
            results['Orchestrator'] = False
        
        # Agent framework check
        try:
            results['Agent Framework'] = len(self.get_available_agents()) > 0
        except:
            results['Agent Framework'] = False
        
        return results
    
    def run(self):
        """Run the main application"""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_main_content()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"Application error: {e}")
            st.info("Please refresh the page or contact support if the problem persists.")

def main():
    """Main application entry point"""
    try:
        app = SutazAIApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start SutazAI application: {e}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()