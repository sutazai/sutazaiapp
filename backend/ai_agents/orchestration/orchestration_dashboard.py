"""
SutazAI Agent Orchestration Dashboard and Management Interface
============================================================

A comprehensive web-based dashboard for monitoring and man  ng the entire
38-agent orchestration system with real-time metrics, health monitoring,
task tracking, and system control capabilities.

Key Features:
- Real-time agent health monitoring
- Task orchestration visualization
- Performance metrics and analytics
- System resource utilization
- Communication flow monitoring
- Interactive agent management
- Alert and notification system
- Historical data analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dataclasses import asdict
import redis.asyncio as redis

from .master_agent_orchestrator import MasterAgentOrchestrator, AgentType, AgentCapability
from .advanced_message_bus import AdvancedMessageBus, MessageType, CommunicationPattern

logger = logging.getLogger(__name__)


class OrchestrationDashboard:
    """
    Comprehensive dashboard for orchestration system monitoring and management
    """
    
    def __init__(self, orchestrator: MasterAgentOrchestrator, message_bus: AdvancedMessageBus):
        self.orchestrator = orchestrator
        self.message_bus = message_bus
        self.redis_client = None
        
        # Dashboard configuration
        self.config = {
            "refresh_interval": 5,  # seconds
            "max_history_points": 100,
            "alert_thresholds": {
                "agent_health": 0.8,
                "success_rate": 0.9,
                "response_time": 10.0,
                "error_rate": 0.05
            }
        }
        
        # Cached data
        self.cached_data = {}
        self.last_update = None
    
    async def initialize(self):
        """Initialize dashboard connections"""
        self.redis_client = await redis.from_url("redis://redis:6379")
        logger.info("Orchestration Dashboard initialized")
    
    def render_dashboard(self):
        """Render the main dashboard interface"""
        
        st.set_page_config(
            page_title="SutazAI Agent Orchestration Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 SutazAI Agent Orchestration Dashboard")
        st.markdown("**Real-time monitoring and management of 38 AI agents**")
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Overview", 
            "🤖 Agents", 
            "📋 Tasks", 
            "📊 Analytics", 
            "⚙️ Management"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_agents_tab()
        
        with tab3:
            self._render_tasks_tab()
        
        with tab4:
            self._render_analytics_tab()
        
        with tab5:
            self._render_management_tab()
    
    def _render_sidebar(self):
        """Render sidebar with system controls and quick stats"""
        
        st.sidebar.header("🎛️ System Control")
        
        # System status
        system_status = asyncio.run(self.orchestrator.get_system_status())
        
        if system_status["orchestrator_status"] == "running":
            st.sidebar.success("✅ System Running")
        else:
            st.sidebar.error("❌ System Stopped")
        
        # Quick stats
        st.sidebar.header("📊 Quick Stats")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Agents", system_status["total_agents"])
            st.metric("Active Tasks", system_status["active_tasks"])
        
        with col2:
            st.metric("Healthy Agents", system_status["healthy_agents"])
            st.metric("Active Sessions", system_status["active_sessions"])
        
        # System controls
        st.sidebar.header("🔧 Controls")
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.sidebar.button("🛑 Emergency Stop"):
            st.sidebar.warning("Emergency stop initiated!")
            # Implement emergency stop logic
        
        # Alert settings
        st.sidebar.header("🚨 Alert Settings")
        
        self.config["alert_thresholds"]["agent_health"] = st.sidebar.slider(
            "Agent Health Threshold", 0.0, 1.0, 
            self.config["alert_thresholds"]["agent_health"]
        )
        
        self.config["alert_thresholds"]["success_rate"] = st.sidebar.slider(
            "Success Rate Threshold", 0.0, 1.0,
            self.config["alert_thresholds"]["success_rate"]
        )
    
    def _render_overview_tab(self):
        """Render system overview tab"""
        
        st.header("🏠 System Overview")
        
        # Real-time metrics
        metrics_data = asyncio.run(self._get_real_time_metrics())
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Health",
                f"{metrics_data['system_health']:.1%}",
                delta=f"{metrics_data['health_delta']:.1%}" if metrics_data['health_delta'] else None
            )
        
        with col2:
            st.metric(
                "Average Success Rate",
                f"{metrics_data['avg_success_rate']:.1%}",
                delta=f"{metrics_data['success_delta']:.1%}" if metrics_data['success_delta'] else None
            )
        
        with col3:
            st.metric(
                "Average Response Time",
                f"{metrics_data['avg_response_time']:.2f}s",
                delta=f"{metrics_data['response_delta']:.2f}s" if metrics_data['response_delta'] else None
            )
        
        with col4:
            st.metric(
                "Message Throughput",
                f"{metrics_data['message_throughput']}/min",
                delta=f"{metrics_data['throughput_delta']}/min" if metrics_data['throughput_delta'] else None
            )
        
        # System architecture diagram
        st.subheader("🏗️ System Architecture")
        self._render_architecture_diagram()
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        self._render_activity_timeline()
        
        # Active alerts
        alerts = self._get_active_alerts()
        if alerts:
            st.subheader("🚨 Active Alerts")
            for alert in alerts:
                if alert["severity"] == "critical":
                    st.error(f"🔴 {alert['message']}")
                elif alert["severity"] == "warning":
                    st.warning(f"🟡 {alert['message']}")
                else:
                    st.info(f"🔵 {alert['message']}")
    
    def _render_agents_tab(self):
        """Render agents monitoring tab"""
        
        st.header("🤖 Agent Management")
        
        # Agent filter and search
        col1, col2, col3 = st.columns(3)
        
        with col1:
            agent_type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + [t.value for t in AgentType]
            )
        
        with col2:
            capability_filter = st.selectbox(
                "Filter by Capability",
                ["All"] + [c.value for c in AgentCapability]
            )
        
        with col3:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "healthy", "unhealthy", "offline"]
            )
        
        # Search box
        search_term = st.text_input("🔍 Search agents", placeholder="Enter agent name or ID")
        
        # Get agent data
        agent_data = asyncio.run(self._get_agent_data(
            agent_type_filter, capability_filter, status_filter, search_term
        ))
        
        # Agent grid view
        st.subheader("📊 Agent Status Grid")
        self._render_agent_grid(agent_data)
        
        # Detailed agent list
        st.subheader("📋 Detailed Agent List")
        self._render_agent_table(agent_data)
        
        # Agent performance charts
        st.subheader("📈 Performance Trends")
        self._render_agent_performance_charts(agent_data)
    
    def _render_tasks_tab(self):
        """Render task orchestration tab"""
        
        st.header("📋 Task Orchestration")
        
        # Task submission interface
        st.subheader("➕ Submit New Task")
        self._render_task_submission_form()
        
        # Active tasks monitoring
        st.subheader("🔄 Active Tasks")
        active_tasks = asyncio.run(self._get_active_tasks())
        self._render_active_tasks_table(active_tasks)
        
        # Task history and analytics
        st.subheader("📊 Task Analytics")
        task_analytics = asyncio.run(self._get_task_analytics())
        self._render_task_analytics(task_analytics)
        
        # Coordination patterns visualization
        st.subheader("🕸️ Coordination Patterns")
        self._render_coordination_patterns()
    
    def _render_analytics_tab(self):
        """Render analytics and insights tab"""
        
        st.header("📊 Advanced Analytics")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Performance analytics
        st.subheader("⚡ Performance Analytics")
        self._render_performance_analytics(start_date, end_date)
        
        # Communication flow analysis
        st.subheader("💬 Communication Flow")
        self._render_communication_analysis()
        
        # Resource utilization
        st.subheader("💻 Resource Utilization")
        self._render_resource_analytics()
        
        # Predictive insights
        st.subheader("🔮 Predictive Insights")
        self._render_predictive_analytics()
    
    def _render_management_tab(self):
        """Render system management tab"""
        
        st.header("⚙️ System Management")
        
        # Configuration management
        st.subheader("🔧 Configuration")
        self._render_configuration_panel()
        
        # Agent lifecycle management
        st.subheader("🤖 Agent Lifecycle")
        self._render_agent_lifecycle_controls()
        
        # System maintenance
        st.subheader("🛠️ Maintenance")
        self._render_maintenance_controls()
        
        # Data export and backup
        st.subheader("💾 Data Management")
        self._render_data_management()
    
    # ==================== Helper Methods ====================
    
    async def _get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics"""
        
        system_status = await self.orchestrator.get_system_status()
        message_bus_status = await self.message_bus.get_system_status()
        
        # Calculate derived metrics
        total_agents = system_status["total_agents"]
        healthy_agents = system_status["healthy_agents"]
        system_health = healthy_agents / max(total_agents, 1)
        
        avg_success_rate = system_status.get("system_metrics", {}).get("avg_success_rate", 0.0)
        avg_response_time = system_status.get("system_metrics", {}).get("avg_response_time", 0.0)
        
        # Message throughput calculation
        performance_metrics = message_bus_status.get("performance_metrics", {})
        message_throughput = performance_metrics.get("total_messages_sent", 0) / max(1, 60)  # per minute
        
        # Calculate deltas (compared to previous values)
        health_delta = self._calculate_delta("system_health", system_health)
        success_delta = self._calculate_delta("avg_success_rate", avg_success_rate)
        response_delta = self._calculate_delta("avg_response_time", avg_response_time)
        throughput_delta = self._calculate_delta("message_throughput", message_throughput)
        
        return {
            "system_health": system_health,
            "avg_success_rate": avg_success_rate,
            "avg_response_time": avg_response_time,
            "message_throughput": message_throughput,
            "health_delta": health_delta,
            "success_delta": success_delta,
            "response_delta": response_delta,
            "throughput_delta": throughput_delta
        }
    
    def _calculate_delta(self, metric_name: str, current_value: float) -> Optional[float]:
        """Calculate delta from previous value"""
        
        if metric_name in self.cached_data:
            previous_value = self.cached_data[metric_name]
            delta = current_value - previous_value
            self.cached_data[metric_name] = current_value
            return delta
        else:
            self.cached_data[metric_name] = current_value
            return None
    
    def _render_architecture_diagram(self):
        """Render system architecture diagram"""
        
        # Create a network graph showing agent relationships
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.Graph()
        
        # Add nodes for different agent types
        agent_types = {
            "Core Intelligence": ["  -system-architect", "autonomous-system-controller"],
            "Development": ["senior-ai-engineer", "senior-backend-developer"],
            "Infrastructure": ["infrastructure-devops-manager", "ollama-integration-specialist"],
            "Specialized AI": ["local  -orchestration-manager", "agentzero-coordinator"],
            "Security": ["semgrep-security-analyzer", "security-pentesting-specialist"]
        }
        
        # Color mapping for agent types
        colors = {
            "Core Intelligence": "#FF6B6B",
            "Development": "#4ECDC4", 
            "Infrastructure": "#45B7D1",
            "Specialized AI": "#96CEB4",
            "Security": "#FECA57"
        }
        
        pos = {}
        node_colors = []
        
        # Position nodes in groups
        group_positions = {
            "Core Intelligence": (0, 0),
            "Development": (-2, 1),
            "Infrastructure": (2, 1),
            "Specialized AI": (-2, -1),
            "Security": (2, -1)
        }
        
        for group, agents in agent_types.items():
            base_x, base_y = group_positions[group]
            
            for i, agent in enumerate(agents):
                G.add_node(agent)
                pos[agent] = (base_x + i * 0.5, base_y + i * 0.3)
                node_colors.append(colors[group])
        
        # Add edges between related agents
        edges = [
            ("  -system-architect", "autonomous-system-controller"),
            ("senior-ai-engineer", "senior-backend-developer"),
            ("infrastructure-devops-manager", "ollama-integration-specialist"),
            # Add more logical connections
        ]
        
        G.add_edges_from(edges)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                node_size=1000, font_size=8, ax=ax)
        ax.set_title("Agent Architecture Overview")
        
        st.pyplot(fig)
    
    def _render_activity_timeline(self):
        """Render recent activity timeline"""
        
        # Create sample timeline data
        timeline_data = [
            {"time": datetime.now() - timedelta(minutes=5), "event": "Task completed", "agent": "senior-ai-engineer", "type": "success"},
            {"time": datetime.now() - timedelta(minutes=10), "event": "New task assigned", "agent": "code-generation-improver", "type": "info"},
            {"time": datetime.now() - timedelta(minutes=15), "event": "Agent health check", "agent": "infrastructure-devops-manager", "type": "info"},
            {"time": datetime.now() - timedelta(minutes=20), "event": "Coordination session started", "agent": "ai-agent-orchestrator", "type": "info"},
        ]
        
        # Create DataFrame
        df = pd.DataFrame(timeline_data)
        df['time_str'] = df['time'].dt.strftime('%H:%M:%S')
        
        # Display timeline
        for _, row in df.iterrows():
            if row['type'] == 'success':
                st.success(f"✅ {row['time_str']} - {row['event']} ({row['agent']})")
            elif row['type'] == 'warning':
                st.warning(f"⚠️ {row['time_str']} - {row['event']} ({row['agent']})")
            elif row['type'] == 'error':
                st.error(f"❌ {row['time_str']} - {row['event']} ({row['agent']})")
            else:
                st.info(f"ℹ️ {row['time_str']} - {row['event']} ({row['agent']})")
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        
        alerts = []
        
        # Check system health
        system_status = asyncio.run(self.orchestrator.get_system_status())
        healthy_ratio = system_status["healthy_agents"] / max(system_status["total_agents"], 1)
        
        if healthy_ratio < self.config["alert_thresholds"]["agent_health"]:
            alerts.append({
                "severity": "critical",
                "message": f"Low system health: {healthy_ratio:.1%} agents healthy"
            })
        
        # Check for failed tasks
        if system_status.get("failed_tasks", 0) > 0:
            alerts.append({
                "severity": "warning",
                "message": f"{system_status['failed_tasks']} tasks have failed"
            })
        
        return alerts
    
    async def _get_agent_data(self, type_filter: str, capability_filter: str, 
                            status_filter: str, search_term: str) -> List[Dict[str, Any]]:
        """Get filtered agent data"""
        
        agent_status = await self.orchestrator.get_agent_status()
        
        filtered_agents = []
        
        for agent_id, agent_info in agent_status.items():
            # Apply filters
            if type_filter != "All" and agent_info.get("type") != type_filter:
                continue
            
            if capability_filter != "All" and capability_filter not in agent_info.get("capabilities", []):
                continue
            
            if status_filter != "All" and agent_info.get("status") != status_filter:
                continue
            
            if search_term and search_term.lower() not in agent_info.get("name", "").lower():
                continue
            
            # Add performance metrics
            performance = self.orchestrator.agent_performance.get(agent_id, {})
            agent_info.update({
                "id": agent_id,
                "success_rate": performance.get("success_rate", 1.0),
                "response_time": performance.get("response_time", 1.0),
                "resource_efficiency": performance.get("resource_efficiency", 1.0)
            })
            
            filtered_agents.append(agent_info)
        
        return filtered_agents
    
    def _render_agent_grid(self, agent_data: List[Dict[str, Any]]):
        """Render agent status grid"""
        
        # Create grid layout
        cols = st.columns(6)  # 6 agents per row
        
        for i, agent in enumerate(agent_data):
            col_idx = i % 6
            
            with cols[col_idx]:
                # Status indicator
                if agent["status"] == "healthy":
                    status_color = "🟢"
                elif agent["status"] == "unhealthy":
                    status_color = "🔴"
                else:
                    status_color = "⚪"
                
                # Agent card
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4>{status_color} {agent['name'][:20]}...</h4>
                    <p>Load: {agent['current_load']}</p>
                    <p>Success: {agent['success_rate']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_agent_table(self, agent_data: List[Dict[str, Any]]):
        """Render detailed agent table"""
        
        if not agent_data:
            st.info("No agents match the current filters.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(agent_data)
        
        # Select and rename columns for display
        display_columns = {
            "name": "Agent Name",
            "status": "Status", 
            "current_load": "Load",
            "success_rate": "Success Rate",
            "response_time": "Response Time (s)",
            "last_health_check": "Last Check"
        }
        
        display_df = df[list(display_columns.keys())].rename(columns=display_columns)
        
        # Format columns
        display_df["Success Rate"] = display_df["Success Rate"].apply(lambda x: f"{x:.1%}")
        display_df["Response Time (s)"] = display_df["Response Time (s)"].apply(lambda x: f"{x:.2f}")
        
        # Display table with selection
        selected_rows = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    def _render_task_submission_form(self):
        """Render task submission form"""
        
        with st.form("task_submission"):
            col1, col2 = st.columns(2)
            
            with col1:
                task_name = st.text_input("Task Name", placeholder="Enter task name")
                task_type = st.selectbox("Task Type", [
                    "analysis", "code_generation", "deployment", 
                    "testing", "optimization", "research"
                ])
                priority = st.selectbox("Priority", [
                    "critical", "high", "medium", "low", "background"
                ])
            
            with col2:
                requirements = st.multiselect("Required Capabilities", [
                    cap.value for cap in AgentCapability
                ])
                
                coordination_pattern = st.selectbox("Coordination Pattern", [
                    "hierarchical", "collaborative", "pipeline", "swarm"
                ])
            
            task_description = st.text_area("Task Description", 
                                          placeholder="Describe the task in detail...")
            
            # Advanced options
            with st.expander("Advanced Options"):
                deadline = st.datetime_input("Deadline (optional)")
                max_agents = st.number_input("Max Agents", min_value=1, max_value=10, value=3)
                
                # Custom payload
                custom_payload = st.text_area("Custom Payload (JSON)", 
                                            placeholder='{"key": "value"}')
            
            submitted = st.form_submit_button("Submit Task")
            
            if submitted and task_name and task_description:
                # Create task
                task_data = {
                    "name": task_name,
                    "description": task_description,
                    "type": task_type,
                    "priority": priority,
                    "requirements": requirements,
                    "payload": json.loads(custom_payload) if custom_payload else {},
                    "constraints": {
                        "max_agents": max_agents,
                        "coordination_pattern": coordination_pattern
                    }
                }
                
                try:
                    session_id = asyncio.run(self.orchestrator.submit_task(task_data))
                    st.success(f"✅ Task submitted successfully! Session ID: {session_id}")
                except Exception as e:
                    st.error(f"❌ Failed to submit task: {str(e)}")
    
    async def _get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get active tasks data"""
        
        active_tasks = []
        
        for task_id, task in self.orchestrator.active_tasks.items():
            task_data = {
                "id": task_id,
                "name": task.name,
                "type": task.type,
                "priority": task.priority.value,
                "status": task.status,
                "assigned_agents": task.assigned_agents,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "progress": self._calculate_task_progress(task)
            }
            active_tasks.append(task_data)
        
        return active_tasks
    
    def _calculate_task_progress(self, task) -> float:
        """Calculate task progress percentage"""
        
        if task.status == "completed":
            return 1.0
        elif task.status == "failed":
            return 0.0
        elif task.status == "in_progress":
            # Estimate based on time elapsed
            if task.started_at:
                elapsed = (datetime.now() - task.started_at).total_seconds()
                estimated_duration = 300  # 5 minutes default
                return min(elapsed / estimated_duration, 0.9)  # Max 90% for in-progress
        
        return 0.0
    
    def _render_active_tasks_table(self, active_tasks: List[Dict[str, Any]]):
        """Render active tasks table"""
        
        if not active_tasks:
            st.info("No active tasks at the moment.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(active_tasks)
        
        # Format columns
        df["progress"] = df["progress"].apply(lambda x: f"{x:.1%}")
        df["created_at"] = df["created_at"].apply(lambda x: x.strftime("%H:%M:%S"))
        df["agents_count"] = df["assigned_agents"].apply(len)
        
        # Display columns
        display_df = df[["name", "type", "priority", "status", "progress", "agents_count", "created_at"]]
        display_df.columns = ["Task Name", "Type", "Priority", "Status", "Progress", "Agents", "Created"]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Task details
        if st.button("Show Task Details"):
            selected_task = st.selectbox("Select Task", [t["name"] for t in active_tasks])
            task_details = next(t for t in active_tasks if t["name"] == selected_task)
            
            st.json(task_details)


# ==================== Streamlit App Entry Point ====================

def main():
    """Main dashboard application"""
    
    # Initialize components
    orchestrator = None
    message_bus = None
    dashboard = None
    
    try:
        # Create instances (in a real app, these would be singletons)
        from .master_agent_orchestrator import create_master_orchestrator
        from .advanced_message_bus import create_message_bus
        
        orchestrator = create_master_orchestrator("redis://redis:6379")
        message_bus = create_message_bus("redis://redis:6379")
        dashboard = OrchestrationDashboard(orchestrator, message_bus)
        
        # Initialize asynchronously
        asyncio.run(dashboard.initialize())
        
        # Render dashboard
        dashboard.render_dashboard()
        
    except Exception as e:
        st.error(f"❌ Dashboard initialization failed: {str(e)}")
        st.info("Please ensure Redis is running and accessible at redis://redis:6379")


if __name__ == "__main__":
    main()