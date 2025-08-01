#!/usr/bin/env python3
"""
SutazAI Performance Dashboard
Standalone Streamlit dashboard for real-time performance monitoring
"""

import streamlit as st
import requests
import psutil
import time
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure Streamlit
st.set_page_config(
    page_title="SutazAI Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "monitoring_data" not in st.session_state:
    st.session_state.monitoring_data = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 5

def collect_system_metrics():
    """Collect current system performance metrics"""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        # Process count
        process_count = len(psutil.pids())
        
        return {
            "timestamp": datetime.now(),
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_used_gb": disk.used / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
            "network_sent_mb": network.bytes_sent / (1024**2),
            "network_recv_mb": network.bytes_recv / (1024**2),
            "process_count": process_count
        }
    except Exception as e:
        st.error(f"Error collecting system metrics: {e}")
        return None

def check_backend_health():
    """Check if the SutazAI backend is healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "response_time": response.elapsed.total_seconds()}
        else:
            return {"status": "unhealthy", "response_time": None}
    except:
        return {"status": "offline", "response_time": None}

def check_ollama_health():
    """Check if Ollama is healthy"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "version": response.json().get("version", "unknown")}
        else:
            return {"status": "unhealthy", "version": None}
    except:
        return {"status": "offline", "version": None}

def check_docker_services():
    """Check Docker services status"""
    services = {
        "PostgreSQL": {"port": 5432, "host": "localhost"},
        "Redis": {"port": 6379, "host": "localhost"},
        "Qdrant": {"port": 6333, "host": "localhost"},
        "ChromaDB": {"port": 8001, "host": "localhost"}
    }
    
    status = {}
    for service, config in services.items():
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((config["host"], config["port"]))
            sock.close()
            status[service] = "healthy" if result == 0 else "unhealthy"
        except:
            status[service] = "unknown"
    
    return status

def display_system_overview():
    """Display system overview metrics"""
    st.header("📊 System Performance Overview")
    
    # Get current metrics
    metrics = collect_system_metrics()
    if not metrics:
        st.error("Unable to collect system metrics")
        return
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="CPU Usage",
            value=f"{metrics['cpu_percent']:.1f}%",
            delta=None
        )
        
        # CPU status color
        if metrics['cpu_percent'] > 90:
            st.error("🔴 CPU Critical")
        elif metrics['cpu_percent'] > 70:
            st.warning("🟡 CPU High")
        else:
            st.success("🟢 CPU Normal")
    
    with col2:
        st.metric(
            label="Memory Usage",
            value=f"{metrics['memory_percent']:.1f}%",
            delta=f"{metrics['memory_used_gb']:.1f}/{metrics['memory_total_gb']:.1f} GB"
        )
        
        # Memory status color
        if metrics['memory_percent'] > 95:
            st.error("🔴 Memory Critical")
        elif metrics['memory_percent'] > 80:
            st.warning("🟡 Memory High")
        else:
            st.success("🟢 Memory Normal")
    
    with col3:
        st.metric(
            label="Disk Usage",
            value=f"{metrics['disk_percent']:.1f}%",
            delta=f"{metrics['disk_used_gb']:.1f}/{metrics['disk_total_gb']:.1f} GB"
        )
        
        # Disk status color
        if metrics['disk_percent'] > 90:
            st.error("🔴 Disk Critical")
        elif metrics['disk_percent'] > 75:
            st.warning("🟡 Disk High")
        else:
            st.success("🟢 Disk Normal")
    
    with col4:
        st.metric(
            label="Active Processes",
            value=metrics['process_count'],
            delta=None
        )
    
    return metrics

def display_service_status():
    """Display service health status"""
    st.header("🔧 Service Health Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Core Services")
        
        # Backend health
        backend_health = check_backend_health()
        if backend_health["status"] == "healthy":
            st.success(f"✅ SutazAI Backend - {backend_health['response_time']:.3f}s")
        elif backend_health["status"] == "unhealthy":
            st.error("❌ SutazAI Backend - Unhealthy")
        else:
            st.error("❌ SutazAI Backend - Offline")
        
        # Ollama health
        ollama_health = check_ollama_health()
        if ollama_health["status"] == "healthy":
            st.success(f"✅ Ollama - v{ollama_health['version']}")
        elif ollama_health["status"] == "unhealthy":
            st.error("❌ Ollama - Unhealthy")
        else:
            st.error("❌ Ollama - Offline")
    
    with col2:
        st.subheader("Docker Services")
        
        # Docker services
        docker_status = check_docker_services()
        for service, status in docker_status.items():
            if status == "healthy":
                st.success(f"✅ {service}")
            elif status == "unhealthy":
                st.error(f"❌ {service}")
            else:
                st.warning(f"⚠️ {service} - Unknown")

def display_real_time_charts():
    """Display real-time performance charts"""
    st.header("📈 Real-Time Performance Charts")
    
    if len(st.session_state.monitoring_data) < 2:
        st.info("Collecting data... Please wait for charts to appear.")
        return
    
    # Prepare data for charts
    df = pd.DataFrame(st.session_state.monitoring_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Disk Usage (%)', 'Network Activity (MB)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # CPU chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_percent'], 
                  name='CPU %', line=dict(color='red')),
        row=1, col=1
    )
    
    # Memory chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_percent'], 
                  name='Memory %', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Disk chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['disk_percent'], 
                  name='Disk %', line=dict(color='green')),
        row=2, col=1
    )
    
    # Network chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['network_sent_mb'], 
                  name='Sent', line=dict(color='purple')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['network_recv_mb'], 
                  name='Received', line=dict(color='orange')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time")
    
    st.plotly_chart(fig, use_container_width=True)

def display_performance_alerts():
    """Display performance alerts"""
    if not st.session_state.monitoring_data:
        return
    
    latest_metrics = st.session_state.monitoring_data[-1]
    alerts = []
    
    # Check for alerts
    if latest_metrics['cpu_percent'] > 90:
        alerts.append({"type": "critical", "metric": "CPU", "value": f"{latest_metrics['cpu_percent']:.1f}%", "message": "CPU usage critically high"})
    elif latest_metrics['cpu_percent'] > 70:
        alerts.append({"type": "warning", "metric": "CPU", "value": f"{latest_metrics['cpu_percent']:.1f}%", "message": "CPU usage high"})
    
    if latest_metrics['memory_percent'] > 95:
        alerts.append({"type": "critical", "metric": "Memory", "value": f"{latest_metrics['memory_percent']:.1f}%", "message": "Memory usage critically high"})
    elif latest_metrics['memory_percent'] > 80:
        alerts.append({"type": "warning", "metric": "Memory", "value": f"{latest_metrics['memory_percent']:.1f}%", "message": "Memory usage high"})
    
    if latest_metrics['disk_percent'] > 90:
        alerts.append({"type": "critical", "metric": "Disk", "value": f"{latest_metrics['disk_percent']:.1f}%", "message": "Disk usage critically high"})
    elif latest_metrics['disk_percent'] > 75:
        alerts.append({"type": "warning", "metric": "Disk", "value": f"{latest_metrics['disk_percent']:.1f}%", "message": "Disk usage high"})
    
    if alerts:
        st.header("⚠️ Performance Alerts")
        for alert in alerts:
            if alert["type"] == "critical":
                st.error(f"🔴 **{alert['metric']}**: {alert['value']} - {alert['message']}")
            else:
                st.warning(f"🟡 **{alert['metric']}**: {alert['value']} - {alert['message']}")
    else:
        st.success("✅ No performance alerts")

def main():
    """Main dashboard function"""
    
    st.title("📊 SutazAI Performance Dashboard")
    st.markdown("*Real-time system monitoring and performance analytics*")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Auto-refresh controls
        st.session_state.auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.selectbox(
                "Refresh Interval",
                options=[1, 2, 5, 10, 30],
                index=2,  # Default to 5 seconds
                format_func=lambda x: f"{x} seconds"
            )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        # Clear data button
        if st.button("🗑️ Clear Data"):
            st.session_state.monitoring_data = []
            st.rerun()
        
        # Data retention
        max_data_points = st.slider("Max Data Points", 10, 500, 100)
        
        # Display current stats
        st.markdown("---")
        st.subheader("📈 Data Stats")
        st.metric("Data Points", len(st.session_state.monitoring_data))
        if st.session_state.monitoring_data:
            st.metric("Monitoring Since", st.session_state.monitoring_data[0]['timestamp'].strftime("%H:%M:%S"))
    
    # Collect current metrics
    current_metrics = collect_system_metrics()
    if current_metrics:
        st.session_state.monitoring_data.append(current_metrics)
        
        # Limit data points
        if len(st.session_state.monitoring_data) > max_data_points:
            st.session_state.monitoring_data = st.session_state.monitoring_data[-max_data_points:]
    
    # Display dashboard sections
    display_system_overview()
    st.markdown("---")
    display_service_status()
    st.markdown("---")
    display_performance_alerts()
    st.markdown("---")
    display_real_time_charts()
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()