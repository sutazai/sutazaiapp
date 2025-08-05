#!/usr/bin/env python3
"""
Security Monitoring Dashboard for SutazAI
Real-time security monitoring and visualization dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import numpy as np

class SecurityDashboard:
    def __init__(self):
        self.databases = {
            'ids': '/opt/sutazaiapp/data/ids_database.db',
            'security_events': '/opt/sutazaiapp/data/security_events.db',
            'incidents': '/opt/sutazaiapp/data/incidents.db'
        }
        
        # Initialize Streamlit page config
        st.set_page_config(
            page_title="SutazAI Security Dashboard",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_data_from_db(self, db_path, query):
        """Load data from SQLite database"""
        try:
            if not Path(db_path).exists():
                return pd.DataFrame()
            
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error loading data from {db_path}: {e}")
            return pd.DataFrame()
    
    def get_security_metrics(self):
        """Get current security metrics"""
        metrics = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'blocked_ips': 0,
            'active_incidents': 0,
            'threat_score_avg': 0.0,
            'response_success_rate': 0.0
        }
        
        # Get alerts from IDS
        if Path(self.databases['ids']).exists():
            alerts_df = self.load_data_from_db(
                self.databases['ids'],
                "SELECT * FROM alerts WHERE timestamp >= datetime('now', '-24 hours')"
            )
            if not alerts_df.empty:
                metrics['total_alerts'] = len(alerts_df)
                metrics['critical_alerts'] = len(alerts_df[alerts_df['severity'] == 'CRITICAL'])
                metrics['blocked_ips'] = len(alerts_df['source_ip'].dropna().unique())
        
        # Get incidents
        if Path(self.databases['incidents']).exists():
            incidents_df = self.load_data_from_db(
                self.databases['incidents'],
                "SELECT * FROM incidents WHERE status IN ('open', 'responded')"
            )
            if not incidents_df.empty:
                metrics['active_incidents'] = len(incidents_df)
                metrics['threat_score_avg'] = incidents_df['threat_score'].mean()
        
        # Get security events
        if Path(self.databases['security_events']).exists():
            events_df = self.load_data_from_db(
                self.databases['security_events'],
                "SELECT * FROM security_events WHERE timestamp >= datetime('now', '-24 hours')"
            )
            if not events_df.empty:
                metrics['threat_score_avg'] = events_df['threat_score'].mean()
        
        return metrics
    
    def create_metrics_cards(self, metrics):
        """Create metric cards for the dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Alerts (24h)",
                value=metrics['total_alerts'],
                delta=None
            )
        
        with col2:
            st.metric(
                label="Critical Alerts",
                value=metrics['critical_alerts'],
                delta=None
            )
        
        with col3:
            st.metric(
                label="Active Incidents",
                value=metrics['active_incidents'],
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Threat Score",
                value=f"{metrics['threat_score_avg']:.1f}",
                delta=None
            )
    
    def create_alerts_timeline(self):
        """Create alerts timeline visualization"""
        st.subheader("Security Alerts Timeline")
        
        # Load alert data
        alerts_df = self.load_data_from_db(
            self.databases['ids'],
            """
            SELECT timestamp, severity, type, source_ip, description
            FROM alerts 
            WHERE timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
            """
        )
        
        if alerts_df.empty:
            st.info("No alert data available")
            return
        
        # Convert timestamp to datetime
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Create timeline chart
        fig = px.scatter(
            alerts_df,
            x='timestamp',
            y='severity',
            color='severity',
            size_max=15,
            hover_data=['type', 'source_ip', 'description'],
            title="Security Alerts Over Time",
            color_discrete_map={
                'CRITICAL': '#FF0000',
                'HIGH': '#FF6600',
                'MEDIUM': '#FFAA00',
                'LOW': '#00AA00'
            }
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_threat_heatmap(self):
        """Create threat source heatmap"""
        st.subheader("Threat Sources Heatmap")
        
        # Load security events data
        events_df = self.load_data_from_db(
            self.databases['security_events'],
            """
            SELECT source_ip, event_type, threat_score, timestamp
            FROM security_events 
            WHERE source_ip IS NOT NULL 
            AND timestamp >= datetime('now', '-24 hours')
            """
        )
        
        if events_df.empty:
            st.info("No threat data available")
            return
        
        # Aggregate threat data by IP and event type
        threat_matrix = events_df.groupby(['source_ip', 'event_type'])['threat_score'].mean().unstack(fill_value=0)
        
        if threat_matrix.empty:
            st.info("No threat matrix data available")
            return
        
        # Create heatmap
        fig = px.imshow(
            threat_matrix.values,
            x=threat_matrix.columns,
            y=threat_matrix.index,
            color_continuous_scale='Reds',
            title="Threat Score Heatmap by Source IP and Event Type"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_incident_status_chart(self):
        """Create incident status pie chart"""
        st.subheader("Incident Status Distribution")
        
        incidents_df = self.load_data_from_db(
            self.databases['incidents'],
            """
            SELECT status, COUNT(*) as count
            FROM incidents 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY status
            """
        )
        
        if incidents_df.empty:
            st.info("No incident data available")
            return
        
        # Create pie chart
        fig = px.pie(
            incidents_df,
            values='count',
            names='status',
            title="Incident Status Distribution (7 days)",
            color_discrete_map={
                'open': '#FF6B6B',
                'responded': '#4ECDC4',
                'resolved': '#45B7D1',
                'escalated': '#FFA07A'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    def create_network_activity_chart(self):
        """Create network activity monitoring chart"""
        st.subheader("Network Activity Monitoring")
        
        # Get network statistics (simulated data for demo)
        try:
            # Get current network connections
            result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                listening_ports = []
                for line in lines:
                    if 'LISTEN' in line and '0.0.0.0:' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            port = parts[3].split(':')[-1]
                            if port.isdigit():
                                listening_ports.append(int(port))
                
                if listening_ports:
                    port_counts = pd.Series(listening_ports).value_counts().head(10)
                    
                    fig = px.bar(
                        x=port_counts.index,
                        y=port_counts.values,
                        title="Top 10 Listening Ports",
                        labels={'x': 'Port', 'y': 'Count'}
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No listening ports detected")
            else:
                st.warning("Unable to fetch network statistics")
        
        except Exception as e:
            st.error(f"Error getting network statistics: {e}")
    
    def create_threat_trends_chart(self):
        """Create threat trends over time"""
        st.subheader("Threat Trends Analysis")
        
        # Load threat data over time
        events_df = self.load_data_from_db(
            self.databases['security_events'],
            """
            SELECT 
                DATE(timestamp) as date,
                event_type,
                AVG(threat_score) as avg_threat_score,
                COUNT(*) as event_count
            FROM security_events 
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp), event_type
            ORDER BY date
            """
        )
        
        if events_df.empty:
            st.info("No trend data available")
            return
        
        # Convert date column
        events_df['date'] = pd.to_datetime(events_df['date'])
        
        # Create subplot with dual y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Event Count Trends', 'Threat Score Trends'),
            vertical_spacing=0.1
        )
        
        # Event count trends
        for event_type in events_df['event_type'].unique():
            type_data = events_df[events_df['event_type'] == event_type]
            fig.add_trace(
                go.Scatter(
                    x=type_data['date'],
                    y=type_data['event_count'],
                    mode='lines+markers',
                    name=f'{event_type} (Count)',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Threat score trends
        for event_type in events_df['event_type'].unique():
            type_data = events_df[events_df['event_type'] == event_type]
            fig.add_trace(
                go.Scatter(
                    x=type_data['date'],
                    y=type_data['avg_threat_score'],
                    mode='lines+markers',
                    name=f'{event_type} (Score)',
                    line=dict(width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Threat Score", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_response_effectiveness_chart(self):
        """Create automated response effectiveness chart"""
        st.subheader("Automated Response Effectiveness")
        
        # Load response log data
        response_df = self.load_data_from_db(
            self.databases['incidents'],
            """
            SELECT 
                action_type,
                success,
                COUNT(*) as count,
                AVG(execution_time) as avg_execution_time
            FROM response_log 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY action_type, success
            """
        )
        
        if response_df.empty:
            st.info("No response data available")
            return
        
        # Calculate success rates
        success_rates = response_df.groupby('action_type').apply(
            lambda x: x[x['success'] == 1]['count'].sum() / x['count'].sum() * 100
        ).reset_index(name='success_rate')
        
        if not success_rates.empty:
            fig = px.bar(
                success_rates,
                x='action_type',
                y='success_rate',
                title="Response Action Success Rates",
                labels={'success_rate': 'Success Rate (%)', 'action_type': 'Action Type'}
            )
            
            fig.update_layout(height=400)
            fig.update_traces(marker_color='lightblue')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No success rate data available")
    
    def create_top_threats_table(self):
        """Create top threats table"""
        st.subheader("Top Security Threats")
        
        # Load top threats data
        threats_df = self.load_data_from_db(
            self.databases['security_events'],
            """
            SELECT 
                source_ip,
                COUNT(*) as event_count,
                AVG(threat_score) as avg_threat_score,
                MAX(threat_score) as max_threat_score,
                GROUP_CONCAT(DISTINCT event_type) as event_types,
                MAX(timestamp) as last_seen
            FROM security_events 
            WHERE source_ip IS NOT NULL 
            AND timestamp >= datetime('now', '-24 hours')
            GROUP BY source_ip
            HAVING avg_threat_score >= 5.0
            ORDER BY avg_threat_score DESC, event_count DESC
            LIMIT 20
            """
        )
        
        if threats_df.empty:
            st.info("No high-threat sources detected")
            return
        
        # Format the data for display
        threats_df['avg_threat_score'] = threats_df['avg_threat_score'].round(1)
        threats_df['max_threat_score'] = threats_df['max_threat_score'].round(1)
        threats_df['last_seen'] = pd.to_datetime(threats_df['last_seen']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display as interactive table
        st.dataframe(
            threats_df,
            column_config={
                'source_ip': 'Source IP',
                'event_count': 'Events',
                'avg_threat_score': 'Avg Score',
                'max_threat_score': 'Max Score',
                'event_types': 'Event Types',
                'last_seen': 'Last Seen'
            },
            use_container_width=True
        )
    
    def create_system_health_indicators(self):
        """Create system health indicators"""
        st.subheader("Security System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # IDS Status
            ids_status = "🟢 Active" if Path(self.databases['ids']).exists() else "🔴 Inactive"
            st.metric("Intrusion Detection", ids_status)
        
        with col2:
            # Event Logging Status
            logging_status = "🟢 Active" if Path(self.databases['security_events']).exists() else "🔴 Inactive"
            st.metric("Event Logging", logging_status)
        
        with col3:
            # Response System Status
            response_status = "🟢 Active" if Path(self.databases['incidents']).exists() else "🔴 Inactive"
            st.metric("Threat Response", response_status)
    
    def create_real_time_alerts_feed(self):
        """Create real-time alerts feed"""
        st.subheader("Recent Security Alerts")
        
        # Load recent alerts
        recent_alerts_df = self.load_data_from_db(
            self.databases['ids'],
            """
            SELECT timestamp, severity, type, source_ip, description
            FROM alerts 
            ORDER BY timestamp DESC
            LIMIT 20
            """
        )
        
        if recent_alerts_df.empty:
            st.info("No recent alerts")
            return
        
        # Format timestamps
        recent_alerts_df['timestamp'] = pd.to_datetime(recent_alerts_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display alerts with color coding
        for _, alert in recent_alerts_df.iterrows():
            severity_color = {
                'CRITICAL': 'red',
                'HIGH': 'orange',
                'MEDIUM': 'yellow',
                'LOW': 'green'
            }.get(alert['severity'], 'gray')
            
            with st.expander(f"[{alert['severity']}] {alert['type']} - {alert['timestamp']}"):
                st.write(f"**Source IP:** {alert['source_ip'] or 'Unknown'}")
                st.write(f"**Description:** {alert['description']}")
    
    def run_dashboard(self):
        """Run the main dashboard"""
        # Header
        st.title("🔒 SutazAI Security Monitoring Dashboard")
        st.markdown("---")
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Refresh button
        if st.sidebar.button("Refresh Dashboard"):
            st.experimental_rerun()
        
        # Get current metrics
        metrics = self.get_security_metrics()
        
        # Display metrics cards
        self.create_metrics_cards(metrics)
        st.markdown("---")
        
        # System health indicators
        self.create_system_health_indicators()
        st.markdown("---")
        
        # Main dashboard content in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Threats", "Network", "Incidents", "Real-time"
        ])
        
        with tab1:
            # Overview tab
            col1, col2 = st.columns(2)
            with col1:
                self.create_alerts_timeline()
            with col2:
                self.create_incident_status_chart()
            
            self.create_threat_trends_chart()
        
        with tab2:
            # Threats tab
            self.create_threat_heatmap()
            self.create_top_threats_table()
        
        with tab3:
            # Network tab
            self.create_network_activity_chart()
        
        with tab4:
            # Incidents tab
            self.create_response_effectiveness_chart()
        
        with tab5:
            # Real-time tab
            self.create_real_time_alerts_feed()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "SutazAI Security Dashboard | Last updated: " + 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            "</div>",
            unsafe_allow_html=True
        )

def main():
    """Main function to run the dashboard"""
    dashboard = SecurityDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()