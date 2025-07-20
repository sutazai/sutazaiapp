import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def system_monitoring(BACKEND_URL):
    """
    Renders the system monitoring interface.
    """
    st.header("📊 System Metrics & Performance")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
        else:
            st.error(f"Failed to load metrics: {response.status_code}")
            metrics = {}
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Please ensure it is running and accessible.")
        metrics = {}
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        metrics = {}
    
    if metrics:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", metrics.get("total_requests", 0))
        with col2:
            st.metric("Active Agents", metrics.get("active_agents", 0))
        with col3:
            st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
        with col4:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time chart
            st.subheader("Response Time Trend")
            # Mock data for demonstration
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                end=datetime.now(), freq='1H')
            response_times = [0.5 + i * 0.1 for i in range(len(times))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=response_times, mode='lines+markers'))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Agent utilization
            st.subheader("Agent Utilization")
            agents = ["AutoGPT", "LocalAGI", "TabbyML", "Semgrep", "Others"]
            utilization = [30, 25, 20, 15, 10]
            
            fig = px.pie(values=utilization, names=agents, hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
