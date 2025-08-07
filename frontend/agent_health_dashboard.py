#!/usr/bin/env python3
"""
Agent Health Monitoring Dashboard for SutazAI
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="SutazAI Agent Health Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .status-passed {
        background-color: #d4edda;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Load agent data
@st.cache_data
def load_agent_data():
    """Load agent validation and capability data"""
    base_path = Path("/opt/sutazaiapp")
    
    # Load validation report
    validation_report_path = base_path / "agent_validation_report.json"
    if validation_report_path.exists():
        with open(validation_report_path, 'r') as f:
            validation_data = json.load(f)
    else:
        validation_data = None
    
    # Load agent registry from .claude/agents
    agents_path = base_path / ".claude" / "agents" / "sutazai_agents.json"
    if agents_path.exists():
        with open(agents_path, 'r') as f:
            registry_data = json.load(f)
    else:
        registry_data = None
    
    return validation_data, registry_data

# Main dashboard
def main():
    st.title("🤖 SutazAI Agent Health Dashboard")
    st.markdown("Real-time monitoring and health status of all AI agents in the system")
    
    # Load data
    validation_data, registry_data = load_agent_data()
    
    if not validation_data:
        st.error("No validation data found. Please run the agent validation script first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Filter options
        st.subheader("Filter Agents")
        status_filter = st.multiselect(
            "Status",
            ["Passed", "Warning", "Failed"],
            default=["Passed", "Warning", "Failed"]
        )
        
        model_filter = st.multiselect(
            "Model Type",
            ["sonnet", "opus", "haiku"],
            default=["sonnet", "opus", "haiku"]
        )
        
        # Last update time
        st.divider()
        st.caption(f"Last updated: {validation_data.get('timestamp', 'Unknown')}")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Summary metrics
    summary = validation_data.get('summary', {})
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Agents</div>
            <div class="metric-value">{summary.get('total', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #d4edda;">
            <div class="metric-label">Healthy</div>
            <div class="metric-value" style="color: #155724;">{summary.get('passed', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #fff3cd;">
            <div class="metric-label">Warnings</div>
            <div class="metric-value" style="color: #856404;">{summary.get('warnings', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8d7da;">
            <div class="metric-label">Failed</div>
            <div class="metric-value" style="color: #721c24;">{summary.get('failed', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Agent Details", "📈 Analytics", "🗺️ Capability Matrix"])
    
    with tab1:
        st.header("System Health Overview")
        
        # Health status pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['Healthy', 'Warning', 'Failed'],
                values=[summary.get('passed', 0), summary.get('warnings', 0), summary.get('failed', 0)],
                hole=.3,
                marker_colors=['#28a745', '#ffc107', '#dc3545']
            )])
            fig.update_layout(title="Agent Health Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Capabilities by model
            agent_registry = validation_data.get('agent_registry', {})
            model_caps = defaultdict(list)
            
            for agent_id, info in agent_registry.items():
                model = info.get('model', 'unknown')
                caps = info.get('capability_count', 0)
                model_caps[model].append(caps)
            
            model_avg_caps = {model: sum(caps)/len(caps) if caps else 0 
                            for model, caps in model_caps.items()}
            
            fig = go.Figure(data=[
                go.Bar(x=list(model_avg_caps.keys()), 
                      y=list(model_avg_caps.values()),
                      marker_color=['#17a2b8', '#6610f2', '#e83e8c'])
            ])
            fig.update_layout(title="Average Capabilities by Model", height=400,
                            xaxis_title="Model", yaxis_title="Avg Capabilities")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Agent Details")
        
        # Search box
        search_term = st.text_input("🔍 Search agents...", placeholder="Type agent name or capability")
        
        # Agent list
        agent_registry = validation_data.get('agent_registry', {})
        results = validation_data.get('results', {})
        
        # Filter agents
        filtered_agents = []
        for agent_id, info in agent_registry.items():
            # Check status
            if agent_id in results.get('passed', []):
                status = 'Passed'
            elif agent_id in results.get('warnings', []):
                status = 'Warning'
            elif agent_id in results.get('failed', []):
                status = 'Failed'
            else:
                status = 'Unknown'
            
            # Apply filters
            if status not in status_filter:
                continue
            
            model = info.get('model', 'unknown')
            if model not in model_filter:
                continue
            
            # Search filter
            if search_term:
                search_lower = search_term.lower()
                if (search_lower not in agent_id.lower() and 
                    not any(search_lower in cap.lower() for cap in info.get('capabilities', []))):
                    continue
            
            filtered_agents.append((agent_id, info, status))
        
        # Display agents
        for agent_id, info, status in sorted(filtered_agents):
            status_class = f"status-{status.lower()}"
            with st.expander(f"{agent_id} - {status}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Model:** {info.get('model', 'unknown')}")
                    st.markdown(f"**Capabilities:** {info.get('capability_count', 0)}")
                    
                    # Show first few capabilities
                    caps = info.get('capabilities', [])[:5]
                    if caps:
                        st.markdown("**Top Capabilities:**")
                        for cap in caps:
                            st.markdown(f"- {cap}")
                        if len(info.get('capabilities', [])) > 5:
                            st.caption(f"...and {len(info.get('capabilities', [])) - 5} more")
                
                with col2:
                    st.markdown(f"""
                    <div class="agent-card {status_class}">
                        <strong>Status:</strong> {status}
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Agent Analytics")
        
        # Capability distribution
        agent_registry = validation_data.get('agent_registry', {})
        cap_counts = [info.get('capability_count', 0) for info in agent_registry.values()]
        
        fig = go.Figure(data=[go.Histogram(x=cap_counts, nbinsx=20)])
        fig.update_layout(
            title="Capability Count Distribution",
            xaxis_title="Number of Capabilities",
            yaxis_title="Number of Agents",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top agents by capabilities
        st.subheader("Top 10 Agents by Capability Count")
        top_agents = sorted(
            [(agent_id, info.get('capability_count', 0)) for agent_id, info in agent_registry.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        df = pd.DataFrame(top_agents, columns=['Agent', 'Capabilities'])
        fig = px.bar(df, x='Capabilities', y='Agent', orientation='h',
                    color='Capabilities', color_continuous_scale='viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Agent Capability Matrix")
        
        # Load capability matrix if exists
        matrix_path = Path("/opt/sutazaiapp/agent_capability_matrix.md")
        if matrix_path.exists():
            with open(matrix_path, 'r') as f:
                matrix_content = f.read()
            
            st.markdown(matrix_content)
        else:
            st.warning("Capability matrix not found. Run the validation script to generate it.")
    
    # Footer
    st.divider()
    st.caption("SutazAI Agent Health Dashboard v1.0 | Auto-refreshes every 5 minutes")

if __name__ == "__main__":
    main()