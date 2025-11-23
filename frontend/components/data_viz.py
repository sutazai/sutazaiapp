"""
Real-Time Data Visualization Dashboards
Provides Plotly charts for system metrics, agent activity, and performance monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import numpy as np


class SystemMetricsDashboard:
    """Real-time system metrics visualization"""
    
    @staticmethod
    def render_resource_usage_chart(cpu_data: List[float], memory_data: List[float],
                                   timestamps: Optional[List[datetime]] = None,
                                   height: int = 400) -> go.Figure:
        """
        Create resource usage chart with CPU and memory
        
        Args:
            cpu_data: List of CPU usage percentages
            memory_data: List of memory usage percentages
            timestamps: Optional list of timestamps
            height: Chart height in pixels
        
        Returns:
            Plotly figure object
        """
        if timestamps is None:
            timestamps = list(range(len(cpu_data)))
        
        fig = go.Figure()
        
        # CPU usage
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cpu_data,
            mode='lines',
            name='CPU Usage',
            line=dict(color='#00D4FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.2)'
        ))
        
        # Memory usage
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=memory_data,
            mode='lines',
            name='Memory Usage',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            title="System Resource Usage",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            height=height,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def render_container_health_grid(containers: List[Dict[str, Any]]) -> go.Figure:
        """
        Create grid visualization of container health status
        
        Args:
            containers: List of container info dictionaries
        
        Returns:
            Plotly figure object
        """
        # Prepare data
        names = [c.get('name', 'Unknown') for c in containers]
        statuses = [c.get('status', 'unknown') for c in containers]
        cpu = [c.get('cpu', 0) for c in containers]
        memory = [c.get('memory', 0) for c in containers]
        
        # Color coding based on status
        colors = []
        for status in statuses:
            if 'running' in status.lower() or 'healthy' in status.lower():
                colors.append('#4CAF50')  # Green
            elif 'starting' in status.lower():
                colors.append('#FF9800')  # Orange
            else:
                colors.append('#F44336')  # Red
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Container Health Status', 'Resource Usage'),
            vertical_spacing=0.15,
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        
        # Health status
        fig.add_trace(
            go.Bar(
                x=names,
                y=[1] * len(names),
                marker_color=colors,
                name='Status',
                text=statuses,
                textposition='inside',
                hovertemplate='%{x}<br>Status: %{text}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Resource usage
        fig.add_trace(
            go.Bar(
                x=names,
                y=cpu,
                name='CPU %',
                marker_color='#00D4FF',
                hovertemplate='%{x}<br>CPU: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=memory,
                name='Memory MB',
                marker_color='#FF6B6B',
                hovertemplate='%{x}<br>Memory: %{y:.0f}MB<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    @staticmethod
    def render_response_time_distribution(response_times: List[float],
                                         bins: int = 30) -> go.Figure:
        """
        Create histogram of response time distribution
        
        Args:
            response_times: List of response times in milliseconds
            bins: Number of histogram bins
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=response_times,
            nbinsx=bins,
            name='Response Time',
            marker_color='#00D4FF',
            opacity=0.7
        ))
        
        # Add median line
        median = np.median(response_times)
        fig.add_vline(
            x=median,
            line_dash="dash",
            line_color="#4CAF50",
            annotation_text=f"Median: {median:.0f}ms",
            annotation_position="top"
        )
        
        # Add 95th percentile
        p95 = np.percentile(response_times, 95)
        fig.add_vline(
            x=p95,
            line_dash="dash",
            line_color="#FF9800",
            annotation_text=f"P95: {p95:.0f}ms",
            annotation_position="top"
        )
        
        fig.update_layout(
            template="plotly_dark",
            title="Response Time Distribution",
            xaxis_title="Response Time (ms)",
            yaxis_title="Frequency",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig


class AgentActivityDashboard:
    """Agent activity and performance visualization"""
    
    @staticmethod
    def render_agent_usage_pie(agent_usage: Dict[str, int]) -> go.Figure:
        """
        Create pie chart of agent usage
        
        Args:
            agent_usage: Dictionary mapping agent names to usage counts
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Pie(
            labels=list(agent_usage.keys()),
            values=list(agent_usage.values()),
            hole=0.4,  # Donut chart
            marker=dict(
                colors=['#00D4FF', '#FF6B6B', '#4CAF50', '#FF9800', 
                       '#9C27B0', '#2196F3', '#FFC107', '#E91E63']
            ),
            textinfo='label+percent',
            hovertemplate='%{label}<br>%{value} requests<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            template="plotly_dark",
            title="Agent Usage Distribution",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    @staticmethod
    def render_agent_performance_radar(agent_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create radar chart comparing agent performance metrics
        
        Args:
            agent_metrics: Dictionary of agent names to their metric dictionaries
                          (e.g., {'speed': 0.8, 'accuracy': 0.9, 'cost': 0.6})
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Metrics categories
        categories = ['Speed', 'Accuracy', 'Cost Efficiency', 'Reliability', 'Versatility']
        
        colors = ['#00D4FF', '#FF6B6B', '#4CAF50', '#FF9800', '#9C27B0']
        
        for i, (agent_name, metrics) in enumerate(agent_metrics.items()):
            values = [
                metrics.get('speed', 0.5),
                metrics.get('accuracy', 0.5),
                metrics.get('cost', 0.5),
                metrics.get('reliability', 0.5),
                metrics.get('versatility', 0.5)
            ]
            
            # Close the radar
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=agent_name,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            template="plotly_dark",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Agent Performance Comparison",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def render_agent_timeline(agent_events: List[Dict[str, Any]]) -> go.Figure:
        """
        Create timeline of agent activities
        
        Args:
            agent_events: List of event dictionaries with timestamp, agent, and action
        
        Returns:
            Plotly figure object
        """
        # Sort events by timestamp
        sorted_events = sorted(agent_events, key=lambda x: x.get('timestamp', ''))
        
        agents = list(set([e.get('agent', 'Unknown') for e in sorted_events]))
        agent_colors = {agent: color for agent, color in zip(
            agents,
            ['#00D4FF', '#FF6B6B', '#4CAF50', '#FF9800', '#9C27B0', '#2196F3', '#FFC107', '#E91E63']
        )}
        
        fig = go.Figure()
        
        for agent in agents:
            agent_events_filtered = [e for e in sorted_events if e.get('agent') == agent]
            
            if agent_events_filtered:
                timestamps = [e.get('timestamp') for e in agent_events_filtered]
                actions = [e.get('action', 'Activity') for e in agent_events_filtered]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=[agent] * len(timestamps),
                    mode='markers+text',
                    name=agent,
                    marker=dict(
                        size=12,
                        color=agent_colors.get(agent, '#00D4FF')
                    ),
                    text=actions,
                    textposition="top center",
                    hovertemplate='%{text}<br>%{x}<extra></extra>'
                ))
        
        fig.update_layout(
            template="plotly_dark",
            title="Agent Activity Timeline",
            xaxis_title="Time",
            yaxis_title="Agent",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest'
        )
        
        return fig


class PerformanceMetrics:
    """Calculate and display performance metrics"""
    
    @staticmethod
    def calculate_uptime_percentage(uptime_seconds: int, total_seconds: int = 86400) -> float:
        """Calculate uptime percentage (default: last 24 hours)"""
        return (uptime_seconds / total_seconds) * 100
    
    @staticmethod
    def calculate_success_rate(successful: int, total: int) -> float:
        """Calculate success rate percentage"""
        if total == 0:
            return 0.0
        return (successful / total) * 100
    
    @staticmethod
    def render_kpi_cards(metrics: Dict[str, Any]):
        """
        Render KPI metric cards
        
        Args:
            metrics: Dictionary of metric names to values
        """
        # Create columns for metrics
        num_metrics = len(metrics)
        cols = st.columns(min(num_metrics, 4))
        
        for i, (name, value) in enumerate(metrics.items()):
            with cols[i % 4]:
                # Determine delta and color
                delta = value.get('delta') if isinstance(value, dict) else None
                actual_value = value.get('value') if isinstance(value, dict) else value
                
                st.metric(
                    label=name,
                    value=actual_value,
                    delta=delta
                )
    
    @staticmethod
    def create_realtime_stream_chart(key: str = "rt_chart") -> go.Figure:
        """
        Create real-time updating chart (placeholder for live data)
        
        Args:
            key: Unique key for session state
        
        Returns:
            Plotly figure object
        """
        # Initialize data in session state if not exists
        if key not in st.session_state:
            st.session_state[key] = {
                "timestamps": [],
                "values": [],
                "max_points": 60
            }
        
        data = st.session_state[key]
        
        # Add new data point (simulated)
        current_time = datetime.now()
        new_value = np.random.randint(40, 80)  # Simulated metric
        
        data["timestamps"].append(current_time)
        data["values"].append(new_value)
        
        # Keep only last N points
        if len(data["timestamps"]) > data["max_points"]:
            data["timestamps"] = data["timestamps"][-data["max_points"]:]
            data["values"] = data["values"][-data["max_points"]:]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data["timestamps"],
            y=data["values"],
            mode='lines+markers',
            name='Real-time Metric',
            line=dict(color='#00D4FF', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            title="Real-Time System Metric",
            xaxis_title="Time",
            yaxis_title="Value",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                range=[
                    data["timestamps"][0] if data["timestamps"] else datetime.now(),
                    data["timestamps"][-1] + timedelta(seconds=5) if data["timestamps"] else datetime.now()
                ]
            )
        )
        
        return fig
