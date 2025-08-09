"""
Dashboard Page Module - Extracted from monolith
SutazAI Main Dashboard with system overview and metrics
"""

import streamlit as st
import asyncio
import httpx
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import sys
import os

# Import shared components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from components.enhanced_ui import ModernMetrics, LoadingComponents, NotificationSystem
from utils.api_client import call_api, handle_api_error
from utils.formatters import format_bytes, format_duration

def show_dashboard():
    """Main dashboard page with system overview and key metrics"""
    
    st.header("🏠 SutazAI Dashboard", divider='rainbow')
    
    # System Status Overview
    with st.container():
        st.subheader("📊 System Status")
        
        # Fetch system health data
        try:
            health_data = asyncio.run(call_api("/health"))
            
            if health_data and handle_api_error(health_data, "health check"):
                # Display system status
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "System Status",
                        health_data.get("status", "unknown").title(),
                        delta="Operational" if health_data.get("status") == "healthy" else "Issues"
                    )
                
                with col2:
                    cache_stats = health_data.get("performance", {}).get("cache_stats", {})
                    hit_rate = cache_stats.get("hit_rate", 0) * 100
                    st.metric("Cache Hit Rate", f"{hit_rate:.1f}%", delta=f"{hit_rate - 50:.1f}%")
                
                with col3:
                    ollama_stats = health_data.get("performance", {}).get("ollama_stats", {})
                    total_requests = ollama_stats.get("total_requests", 0)
                    st.metric("Ollama Requests", total_requests, delta=ollama_stats.get("errors", 0))
                
                with col4:
                    task_stats = health_data.get("performance", {}).get("task_queue_stats", {})
                    completed = task_stats.get("tasks_completed", 0)
                    st.metric("Tasks Completed", completed, delta=task_stats.get("tasks_failed", 0))
                
                # Services status grid
                st.subheader("🔧 Service Health")
                services = health_data.get("services", {})
                
                service_cols = st.columns(len(services))
                for idx, (service, status) in enumerate(services.items()):
                    with service_cols[idx]:
                        status_icon = "✅" if status == "healthy" else "⚠️" if status == "configured" else "❌"
                        st.markdown(f"""
                        <div class="service-card">
                            <h4>{status_icon} {service.title()}</h4>
                            <p>{status.title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            else:
                st.error("Unable to fetch system health data")
                
        except Exception as e:
            st.error(f"Error fetching dashboard data: {str(e)}")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🤖 Start AI Chat", use_container_width=True):
            st.session_state['current_page'] = 'AI Chat'
            st.rerun()
    
    with action_col2:
        if st.button("👥 Manage Agents", use_container_width=True):
            st.session_state['current_page'] = 'Agent Control'
            st.rerun()
    
    with action_col3:
        if st.button("📈 View Analytics", use_container_width=True):
            st.session_state['current_page'] = 'Analytics'
            st.rerun()
    
    # Recent Activity
    st.subheader("📝 Recent Activity")
    
    try:
        # Fetch recent tasks or activities
        activity_data = asyncio.run(call_api("/api/v1/tasks/recent", timeout=3.0))
        
        if activity_data:
            # Display activity log
            for activity in activity_data[:5]:  # Show last 5 activities
                with st.container():
                    activity_time = activity.get('timestamp', 'Unknown')
                    activity_type = activity.get('type', 'General')
                    activity_desc = activity.get('description', 'No description')
                    
                    st.markdown(f"""
                    <div class="activity-item">
                        <span class="activity-time">{activity_time}</span>
                        <span class="activity-type">{activity_type}</span>
                        <span class="activity-desc">{activity_desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No recent activity to display")
            
    except Exception as e:
        st.info("Recent activity unavailable")

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .service-card {
        background: linear-gradient(135deg, rgba(26, 115, 232, 0.1), rgba(106, 75, 162, 0.1));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
    }
    
    .activity-item {
        display: flex;
        gap: 12px;
        padding: 8px 12px;
        margin: 4px 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border-left: 3px solid #1a73e8;
    }
    
    .activity-time {
        color: #888;
        font-size: 0.8em;
        min-width: 120px;
    }
    
    .activity-type {
        font-weight: 600;
        color: #1a73e8;
        min-width: 100px;
    }
    
    .activity-desc {
        color: #333;
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)