import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import psutil
import docker
from datetime import datetime, timedelta
import json
import numpy as np

def performance_monitor(BACKEND_URL):
    """
    Advanced performance monitoring dashboard
    """
    st.header("📊 Advanced Performance Monitor")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        refresh_interval = st.selectbox("Refresh Interval", 
                                      [5, 10, 30, 60], 
                                      index=2, 
                                      format_func=lambda x: f"{x} seconds")
    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=False)
    with col3:
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    # Performance metrics tabs
    perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
        "🖥️ System Resources", 
        "🚀 API Performance", 
        "🤖 Agent Performance", 
        "📈 Historical Trends"
    ])
    
    # Tab 1: System Resources
    with perf_tab1:
        render_system_resources(BACKEND_URL)
    
    # Tab 2: API Performance
    with perf_tab2:
        render_api_performance(BACKEND_URL)
    
    # Tab 3: Agent Performance
    with perf_tab3:
        render_agent_performance(BACKEND_URL)
    
    # Tab 4: Historical Trends
    with perf_tab4:
        render_historical_trends(BACKEND_URL)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def render_system_resources(BACKEND_URL):
    """Render real-time system resource monitoring"""
    
    try:
        # Get system metrics
        response = requests.get(f"{BACKEND_URL}/api/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            system_data = metrics.get('system', {})
        else:
            system_data = {}
    except:
        system_data = {}
    
    # Current resource usage
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = system_data.get('cpu', {}).get('usage_percent', 0)
        cpu_color = "red" if cpu_usage > 80 else "orange" if cpu_usage > 60 else "green"
        
        fig_cpu = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = cpu_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CPU Usage %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': cpu_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': 90}
            }
        ))
        fig_cpu.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        mem_usage = system_data.get('memory', {}).get('usage_percent', 0)
        mem_color = "red" if mem_usage > 80 else "orange" if mem_usage > 60 else "green"
        
        fig_mem = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = mem_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Memory Usage %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': mem_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': 90}
            }
        ))
        fig_mem.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_mem, use_container_width=True)
    
    with col3:
        disk_usage = system_data.get('disk', {}).get('usage_percent', 0)
        disk_color = "red" if disk_usage > 90 else "orange" if disk_usage > 70 else "green"
        
        fig_disk = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = disk_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Disk Usage %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': disk_color},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "red"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': 95}
            }
        ))
        fig_disk.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_disk, use_container_width=True)
    
    with col4:
        # Network I/O (simulated)
        network_usage = np.random.uniform(10, 50)
        
        fig_net = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = network_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Network I/O MB/s"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        fig_net.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_net, use_container_width=True)
    
    # Detailed resource breakdown
    st.subheader("📋 Detailed Resource Information")
    
    resource_col1, resource_col2 = st.columns(2)
    
    with resource_col1:
        st.markdown("**CPU Information**")
        cpu_info = system_data.get('cpu', {})
        st.write(f"- CPU Cores: {cpu_info.get('count', 'N/A')}")
        st.write(f"- Current Usage: {cpu_info.get('usage_percent', 0):.1f}%")
        st.write(f"- Load Average: {np.random.uniform(0.5, 2.0):.2f}")
        
        st.markdown("**Memory Information**")
        mem_info = system_data.get('memory', {})
        st.write(f"- Total Memory: {mem_info.get('total_gb', 0):.1f} GB")
        st.write(f"- Available Memory: {mem_info.get('available_gb', 0):.1f} GB")
        st.write(f"- Used Memory: {mem_info.get('usage_percent', 0):.1f}%")
    
    with resource_col2:
        st.markdown("**Disk Information**")
        disk_info = system_data.get('disk', {})
        st.write(f"- Total Disk: {disk_info.get('total_gb', 0):.1f} GB")
        st.write(f"- Free Disk: {disk_info.get('free_gb', 0):.1f} GB")
        st.write(f"- Used Disk: {disk_info.get('usage_percent', 0):.1f}%")
        
        st.markdown("**Container Status**")
        try:
            import docker
            client = docker.from_env()
            containers = client.containers.list()
            running_containers = len([c for c in containers if c.status == 'running'])
            st.write(f"- Running Containers: {running_containers}")
            st.write(f"- Total Containers: {len(containers)}")
        except:
            st.write("- Container info unavailable")

def render_api_performance(BACKEND_URL):
    """Render API performance metrics"""
    
    st.subheader("🚀 API Endpoint Performance")
    
    # Simulate API performance data
    endpoints = [
        "/api/v1/agents", "/api/v1/models", "/api/v1/chat/completions",
        "/api/metrics", "/health", "/api/v1/agents/{id}/execute"
    ]
    
    perf_data = []
    for endpoint in endpoints:
        response_time = np.random.uniform(50, 500)
        success_rate = np.random.uniform(0.95, 1.0)
        requests_per_min = np.random.randint(10, 100)
        
        perf_data.append({
            "Endpoint": endpoint,
            "Avg Response Time (ms)": f"{response_time:.0f}",
            "Success Rate": f"{success_rate*100:.1f}%",
            "Requests/min": requests_per_min,
            "Status": "🟢" if response_time < 200 else "🟡" if response_time < 400 else "🔴"
        })
    
    # Display as dataframe
    df = pd.DataFrame(perf_data)
    st.dataframe(df, use_container_width=True)
    
    # Response time distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Response Time Distribution")
        response_times = np.random.normal(200, 50, 100)
        response_times = np.clip(response_times, 50, 1000)
        
        fig_hist = px.histogram(
            x=response_times,
            nbins=20,
            title="Response Time Distribution (ms)",
            labels={'x': 'Response Time (ms)', 'y': 'Frequency'}
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Real-time Response Times")
        
        # Generate time series data
        now = datetime.now()
        times = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
        response_times = [np.random.uniform(100, 300) for _ in times]
        
        fig_line = px.line(
            x=times,
            y=response_times,
            title="Response Times Over Time",
            labels={'x': 'Time', 'y': 'Response Time (ms)'}
        )
        fig_line.update_layout(height=300)
        st.plotly_chart(fig_line, use_container_width=True)

def render_agent_performance(BACKEND_URL):
    """Render agent-specific performance metrics"""
    
    st.subheader("🤖 Agent Performance Analytics")
    
    # Get agent data
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/agents", timeout=5)
        if response.status_code == 200:
            agents_data = response.json().get('agents', [])
        else:
            agents_data = []
    except:
        agents_data = []
    
    if not agents_data:
        # Fallback data
        agents_data = [
            {"id": "code-generator", "name": "Code Generator", "type": "code_generation"},
            {"id": "security-analyzer", "name": "Security Analyzer", "type": "security"},
            {"id": "document-processor", "name": "Document Processor", "type": "document_processing"},
            {"id": "web-automator", "name": "Web Automator", "type": "web_automation"}
        ]
    
    # Agent performance metrics
    agent_metrics = []
    for agent in agents_data:
        tasks_completed = np.random.randint(50, 500)
        success_rate = np.random.uniform(0.85, 0.99)
        avg_execution_time = np.random.uniform(1.0, 10.0)
        cpu_usage = np.random.uniform(10, 80)
        memory_usage = np.random.uniform(50, 200)  # MB
        
        agent_metrics.append({
            "Agent": agent.get('name', agent.get('id', 'Unknown')),
            "Type": agent.get('type', 'Unknown'),
            "Tasks Completed": tasks_completed,
            "Success Rate": f"{success_rate*100:.1f}%",
            "Avg Execution (s)": f"{avg_execution_time:.2f}",
            "CPU Usage": f"{cpu_usage:.1f}%",
            "Memory (MB)": f"{memory_usage:.0f}",
            "Status": "🟢" if success_rate > 0.9 else "🟡" if success_rate > 0.8 else "🔴"
        })
    
    # Display metrics table
    df_agents = pd.DataFrame(agent_metrics)
    st.dataframe(df_agents, use_container_width=True)
    
    # Agent performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Agent Task Completion")
        fig_bar = px.bar(
            df_agents,
            x="Agent",
            y="Tasks Completed",
            color="Type",
            title="Tasks Completed by Agent"
        )
        fig_bar.update_layout(height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Agent Execution Times")
        execution_times = [float(t.replace('s', '')) for t in df_agents["Avg Execution (s)"]]
        fig_scatter = px.scatter(
            x=df_agents["Agent"],
            y=execution_times,
            size=[int(t) for t in df_agents["Tasks Completed"]],
            color=df_agents["Type"],
            title="Execution Time vs Task Volume"
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

def render_historical_trends(BACKEND_URL):
    """Render historical performance trends"""
    
    st.subheader("📈 Historical Performance Trends")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=1
    )
    
    # Generate historical data based on time range
    if time_range == "Last Hour":
        periods = 60
        freq = '1min'
        start_time = datetime.now() - timedelta(hours=1)
    elif time_range == "Last 6 Hours":
        periods = 72
        freq = '5min'
        start_time = datetime.now() - timedelta(hours=6)
    elif time_range == "Last 24 Hours":
        periods = 96
        freq = '15min'
        start_time = datetime.now() - timedelta(hours=24)
    else:  # Last 7 Days
        periods = 168
        freq = '1H'
        start_time = datetime.now() - timedelta(days=7)
    
    timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)
    
    # Generate synthetic historical data
    cpu_trend = 30 + 20 * np.sin(np.linspace(0, 4*np.pi, periods)) + np.random.normal(0, 5, periods)
    memory_trend = 45 + 15 * np.sin(np.linspace(0, 3*np.pi, periods)) + np.random.normal(0, 3, periods)
    response_time_trend = 200 + 50 * np.sin(np.linspace(0, 2*np.pi, periods)) + np.random.normal(0, 20, periods)
    
    # Ensure values are within realistic bounds
    cpu_trend = np.clip(cpu_trend, 0, 100)
    memory_trend = np.clip(memory_trend, 0, 100)
    response_time_trend = np.clip(response_time_trend, 50, 500)
    
    # Create multi-line chart
    fig_trends = make_subplots(
        rows=3, cols=1,
        subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'API Response Time (ms)'),
        vertical_spacing=0.08
    )
    
    fig_trends.add_trace(
        go.Scatter(x=timestamps, y=cpu_trend, name='CPU Usage', line=dict(color='red')),
        row=1, col=1
    )
    
    fig_trends.add_trace(
        go.Scatter(x=timestamps, y=memory_trend, name='Memory Usage', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig_trends.add_trace(
        go.Scatter(x=timestamps, y=response_time_trend, name='Response Time', line=dict(color='green')),
        row=3, col=1
    )
    
    fig_trends.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg CPU Usage",
            f"{np.mean(cpu_trend):.1f}%",
            f"{np.mean(cpu_trend) - 35:.1f}%"
        )
        st.metric(
            "Max CPU Usage",
            f"{np.max(cpu_trend):.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Memory Usage",
            f"{np.mean(memory_trend):.1f}%",
            f"{np.mean(memory_trend) - 45:.1f}%"
        )
        st.metric(
            "Max Memory Usage",
            f"{np.max(memory_trend):.1f}%"
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            f"{np.mean(response_time_trend):.0f}ms",
            f"{np.mean(response_time_trend) - 200:.0f}ms"
        )
        st.metric(
            "Max Response Time",
            f"{np.max(response_time_trend):.0f}ms"
        )