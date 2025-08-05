"""
MLflow Experiment Dashboards and Visualization
Interactive dashboards for experiment tracking and analysis
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from .config import mlflow_config
from .agent_tracker import agent_tracking_manager
from .analysis_tools import experiment_analyzer
from .pipeline_automation import pipeline_manager
from .metrics import mlflow_metrics


logger = logging.getLogger(__name__)


class MLflowDashboard:
    """Main MLflow dashboard with multiple views"""
    
    def __init__(self):
        self.client = MlflowClient(tracking_uri=mlflow_config.tracking_uri)
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = 'Overview'
        
        # Configure Streamlit
        st.set_page_config(
            page_title="SutazAI MLflow Dashboard",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the dashboard application"""
        try:
            # Sidebar navigation
            self._render_sidebar()
            
            # Main content
            if st.session_state.page == 'Overview':
                self._render_overview()
            elif st.session_state.page == 'Experiments':
                self._render_experiments()
            elif st.session_state.page == 'Agent Tracking':
                self._render_agent_tracking()
            elif st.session_state.page == 'Pipeline Management':
                self._render_pipeline_management()
            elif st.session_state.page == 'Model Registry':
                self._render_model_registry()
            elif st.session_state.page == 'Analysis & Comparison':
                self._render_analysis()
            elif st.session_state.page == 'System Health':
                self._render_system_health()
            
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")
    
    def _render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("🧪 SutazAI MLflow")
        st.sidebar.markdown("---")
        
        # Navigation
        pages = [
            'Overview',
            'Experiments', 
            'Agent Tracking',
            'Pipeline Management',
            'Model Registry',
            'Analysis & Comparison',
            'System Health'
        ]
        
        selected_page = st.sidebar.selectbox("Navigate to:", pages, 
                                           index=pages.index(st.session_state.page))
        
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Quick stats
        try:
            stats = self._get_quick_stats()
            st.sidebar.metric("Total Experiments", stats.get('total_experiments', 0))
            st.sidebar.metric("Active Runs", stats.get('active_runs', 0))
            st.sidebar.metric("Tracked Agents", stats.get('tracked_agents', 0))
        except Exception as e:
            st.sidebar.error(f"Stats error: {e}")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    def _get_quick_stats(self) -> Dict[str, int]:
        """Get quick statistics for sidebar"""
        try:
            # Get experiments count
            experiments = self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            total_experiments = len(experiments)
            
            # Get active runs count
            active_runs = 0
            for exp in experiments:
                runs = self.client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="attribute.status = 'RUNNING'",
                    max_results=1000
                )
                active_runs += len(runs)
            
            # Get tracked agents count
            agent_stats = agent_tracking_manager.get_all_tracking_stats()
            tracked_agents = agent_stats.get('total_agents', 0)
            
            return {
                'total_experiments': total_experiments,
                'active_runs': active_runs,
                'tracked_agents': tracked_agents
            }
            
        except Exception as e:
            logger.error(f"Failed to get quick stats: {e}")
            return {'total_experiments': 0, 'active_runs': 0, 'tracked_agents': 0}
    
    def _render_overview(self):
        """Render the overview dashboard"""
        st.title("📊 MLflow System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            overview_data = self._get_overview_data()
            
            with col1:
                st.metric(
                    "Total Experiments",
                    overview_data['total_experiments'],
                    delta=overview_data.get('experiments_delta', 0)
                )
            
            with col2:
                st.metric(
                    "Successful Runs",
                    overview_data['successful_runs'],
                    delta=overview_data.get('runs_delta', 0)
                )
            
            with col3:
                st.metric(
                    "Active Agents",
                    overview_data['active_agents'],
                    delta=overview_data.get('agents_delta', 0)
                )
            
            with col4:
                st.metric(
                    "Models Registered",
                    overview_data['registered_models'],
                    delta=overview_data.get('models_delta', 0)
                )
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Experiments Over Time")
                self._render_experiments_timeline()
            
            with col2:
                st.subheader("🎯 Success Rate by Agent Type")
                self._render_success_rate_chart()
            
            # Recent activity
            st.subheader("📝 Recent Activity")
            self._render_recent_activity()
            
        except Exception as e:
            st.error(f"Failed to load overview data: {e}")
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _get_overview_data(self) -> Dict[str, Any]:
        """Get overview dashboard data"""
        try:
            # Get experiments
            experiments = self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            total_experiments = len(experiments)
            
            # Get all runs
            all_runs = []
            for exp in experiments:
                runs = self.client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=1000
                )
                all_runs.extend(runs)
            
            successful_runs = len([r for r in all_runs if r.info.status == 'FINISHED'])
            
            # Get agent stats
            agent_stats = agent_tracking_manager.get_all_tracking_stats()
            active_agents = agent_stats.get('total_agents', 0)
            
            # Mock registered models count (would integrate with Model Registry)
            registered_models = 5  # Placeholder
            
            return {
                'total_experiments': total_experiments,
                'successful_runs': successful_runs,
                'active_agents': active_agents,
                'registered_models': registered_models,
                'all_runs': all_runs
            }
            
        except Exception as e:
            logger.error(f"Failed to get overview data: {e}")
            return {
                'total_experiments': 0,
                'successful_runs': 0,
                'active_agents': 0,
                'registered_models': 0,
                'all_runs': []
            }
    
    def _render_experiments_timeline(self):
        """Render experiments timeline chart"""
        try:
            overview_data = self._get_overview_data()
            runs = overview_data.get('all_runs', [])
            
            if not runs:
                st.info("No runs data available")
                return
            
            # Process runs data
            run_dates = []
            for run in runs:
                if run.info.start_time:
                    run_date = datetime.fromtimestamp(run.info.start_time / 1000).date()
                    run_dates.append(run_date)
            
            if run_dates:
                # Create daily counts
                date_counts = pd.Series(run_dates).value_counts().sort_index()
                
                # Create timeline chart
                fig = px.line(
                    x=date_counts.index,
                    y=date_counts.values,
                    title="Daily Experiment Runs"
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Runs",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data available")
                
        except Exception as e:
            st.error(f"Timeline chart error: {e}")
    
    def _render_success_rate_chart(self):
        """Render success rate by agent type chart"""
        try:
            overview_data = self._get_overview_data()
            runs = overview_data.get('all_runs', [])
            
            if not runs:
                st.info("No runs data available")
                return
            
            # Group by agent type
            agent_stats = {}
            for run in runs:
                agent_type = run.data.tags.get('agent_type', 'Unknown')
                
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {'total': 0, 'successful': 0}
                
                agent_stats[agent_type]['total'] += 1
                if run.info.status == 'FINISHED':
                    agent_stats[agent_type]['successful'] += 1
            
            # Calculate success rates
            agent_types = []
            success_rates = []
            
            for agent_type, stats in agent_stats.items():
                if stats['total'] > 0:
                    success_rate = (stats['successful'] / stats['total']) * 100
                    agent_types.append(agent_type)
                    success_rates.append(success_rate)
            
            if agent_types:
                fig = px.bar(
                    x=agent_types,
                    y=success_rates,
                    title="Success Rate by Agent Type"
                )
                fig.update_layout(
                    xaxis_title="Agent Type",
                    yaxis_title="Success Rate (%)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No agent type data available")
                
        except Exception as e:
            st.error(f"Success rate chart error: {e}")
    
    def _render_recent_activity(self):
        """Render recent activity table"""
        try:
            overview_data = self._get_overview_data()
            runs = overview_data.get('all_runs', [])
            
            if not runs:
                st.info("No recent activity")
                return
            
            # Get most recent runs
            recent_runs = sorted(runs, key=lambda r: r.info.start_time or 0, reverse=True)[:10]
            
            # Create activity table
            activity_data = []
            for run in recent_runs:
                activity_data.append({
                    'Run ID': run.info.run_id[:8] + '...',
                    'Experiment': run.info.experiment_id,
                    'Status': run.info.status,
                    'Agent Type': run.data.tags.get('agent_type', 'Unknown'),
                    'Start Time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M') if run.info.start_time else 'N/A'
                })
            
            if activity_data:
                st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
            else:
                st.info("No recent activity data")
                
        except Exception as e:
            st.error(f"Recent activity error: {e}")
    
    def _render_experiments(self):
        """Render experiments management page"""
        st.title("🧪 Experiments Management")
        
        # Search and filter controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("Search experiments", placeholder="Enter experiment name...")
        
        with col2:
            view_type = st.selectbox("View", ["Active", "Deleted", "All"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Name", "Creation Time", "Last Updated"])
        
        # Get experiments
        try:
            experiments = self._get_experiments_data(search_term, view_type, sort_by)
            
            if experiments:
                # Experiments table
                st.subheader(f"📋 Experiments ({len(experiments)})")
                
                for exp in experiments:
                    with st.expander(f"🧪 {exp['name']} ({exp['total_runs']} runs)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**ID:** {exp['experiment_id']}")
                            st.write(f"**Created:** {exp['creation_time']}")
                            st.write(f"**Total Runs:** {exp['total_runs']}")
                            st.write(f"**Successful Runs:** {exp['successful_runs']}")
                        
                        with col2:
                            if exp['tags']:
                                st.write("**Tags:**")
                                for key, value in exp['tags'].items():
                                    st.write(f"  • {key}: {value}")
                        
                        # Recent runs
                        if exp['recent_runs']:
                            st.write("**Recent Runs:**")
                            runs_df = pd.DataFrame(exp['recent_runs'])
                            st.dataframe(runs_df, use_container_width=True)
                        
                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"View Details", key=f"details_{exp['experiment_id']}"):
                                st.session_state[f"show_details_{exp['experiment_id']}"] = True
                        
                        with col2:
                            if st.button(f"Compare Runs", key=f"compare_{exp['experiment_id']}"):
                                self._render_run_comparison(exp['experiment_id'])
                        
                        with col3:
                            if st.button(f"Export Data", key=f"export_{exp['experiment_id']}"):
                                self._export_experiment_data(exp['experiment_id'])
            else:
                st.info("No experiments found matching the criteria")
                
        except Exception as e:
            st.error(f"Failed to load experiments: {e}")
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def _get_experiments_data(self, search_term: str, view_type: str, sort_by: str) -> List[Dict[str, Any]]:
        """Get experiments data with filtering and sorting"""
        try:
            # Map view type
            view_type_map = {
                "Active": ViewType.ACTIVE_ONLY,
                "Deleted": ViewType.DELETED_ONLY,
                "All": ViewType.ALL
            }
            
            # Get experiments
            experiments = self.client.search_experiments(
                view_type=view_type_map.get(view_type, ViewType.ACTIVE_ONLY)
            )
            
            # Process experiments
            exp_data = []
            for exp in experiments:
                # Apply search filter
                if search_term and search_term.lower() not in exp.name.lower():
                    continue
                
                # Get runs for this experiment
                runs = self.client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=100
                )
                
                successful_runs = len([r for r in runs if r.info.status == 'FINISHED'])
                
                # Get recent runs data
                recent_runs = []
                for run in runs[:5]:  # Latest 5 runs
                    recent_runs.append({
                        'Run ID': run.info.run_id[:8] + '...',
                        'Status': run.info.status,
                        'Start Time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M') if run.info.start_time else 'N/A'
                    })
                
                exp_info = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'creation_time': datetime.fromtimestamp(exp.creation_time / 1000).strftime('%Y-%m-%d %H:%M'),
                    'total_runs': len(runs),
                    'successful_runs': successful_runs,
                    'tags': exp.tags or {},
                    'recent_runs': recent_runs
                }
                
                exp_data.append(exp_info)
            
            # Sort experiments
            if sort_by == "Name":
                exp_data.sort(key=lambda x: x['name'])
            elif sort_by == "Creation Time":
                exp_data.sort(key=lambda x: x['creation_time'], reverse=True)
            
            return exp_data
            
        except Exception as e:
            logger.error(f"Failed to get experiments data: {e}")
            return []
    
    def _render_agent_tracking(self):
        """Render agent tracking page"""
        st.title("🤖 Agent Tracking")
        
        try:
            # Get agent tracking stats
            agent_stats = agent_tracking_manager.get_all_tracking_stats()
            
            # Overview metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Agents", agent_stats.get('total_agents', 0))
            
            with col2:
                active_agents = len([a for a in agent_stats.get('agents', {}).values() 
                                   if a.get('current_run_id')])
                st.metric("Active Runs", active_agents)
            
            with col3:
                total_metrics = sum(a.get('metrics_logged', 0) for a in agent_stats.get('agents', {}).values())
                st.metric("Total Metrics Logged", total_metrics)
            
            # Agents table
            st.subheader("📊 Agent Status")
            
            agents_data = []
            for agent_id, stats in agent_stats.get('agents', {}).items():
                agents_data.append({
                    'Agent ID': agent_id,
                    'Agent Type': stats.get('agent_type', 'Unknown'),
                    'Experiment ID': stats.get('experiment_id', 'N/A'),
                    'Current Run': stats.get('current_run_id', 'None')[:8] + '...' if stats.get('current_run_id') else 'None',
                    'Metrics Logged': stats.get('metrics_logged', 0),
                    'Params Logged': stats.get('params_logged', 0),
                    'Uptime (hrs)': round(stats.get('uptime_seconds', 0) / 3600, 1),
                    'Status': 'Active' if stats.get('current_run_id') else 'Idle'
                })
            
            if agents_data:
                df = pd.DataFrame(agents_data)
                st.dataframe(df, use_container_width=True)
                
                # Agent activity chart
                st.subheader("📈 Agent Activity")
                activity_chart = px.bar(
                    df,
                    x='Agent ID',
                    y='Metrics Logged',
                    color='Agent Type',
                    title="Metrics Logged by Agent"
                )
                st.plotly_chart(activity_chart, use_container_width=True)
            else:
                st.info("No agent tracking data available")
                
        except Exception as e:
            st.error(f"Agent tracking error: {e}")
    
    def _render_pipeline_management(self):
        """Render pipeline management page"""
        st.title("⚙️ ML Pipeline Management")
        
        try:
            # Get pipeline status
            pipelines = pipeline_manager.list_pipelines()
            
            if pipelines:
                st.subheader(f"📋 Pipelines ({len(pipelines)})")
                
                for pipeline in pipelines:
                    with st.expander(f"⚙️ {pipeline['name']} ({'Running' if pipeline['is_running'] else 'Idle'})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Description:** {pipeline['description']}")
                            st.write(f"**Agent ID:** {pipeline['agent_id']}")
                            st.write(f"**Agent Type:** {pipeline['agent_type']}")
                        
                        with col2:
                            st.write(f"**Auto Deploy:** {'Yes' if pipeline['auto_deploy'] else 'No'}")
                            st.write(f"**Hyperparameter Tuning:** {'Yes' if pipeline['hyperparameter_tuning'] else 'No'}")
                            st.write(f"**Status:** {'Running' if pipeline['is_running'] else 'Idle'}")
                        
                        # Pipeline actions
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if not pipeline['is_running']:
                                if st.button(f"▶️ Start", key=f"start_{pipeline['name']}"):
                                    try:
                                        pipeline_manager.execute_pipeline(pipeline['name'])
                                        st.success(f"Started pipeline {pipeline['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to start pipeline: {e}")
                            else:
                                if st.button(f"⏹️ Stop", key=f"stop_{pipeline['name']}"):
                                    try:
                                        pipeline_manager.stop_pipeline(pipeline['name'])
                                        st.success(f"Stopped pipeline {pipeline['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to stop pipeline: {e}")
                        
                        with col2:
                            if st.button(f"📊 View Status", key=f"status_{pipeline['name']}"):
                                status = pipeline_manager.get_pipeline_status(pipeline['name'])
                                st.json(status)
                        
                        with col3:
                            if st.button(f"⚙️ Configure", key=f"config_{pipeline['name']}"):
                                st.info("Pipeline configuration UI would open here")
            else:
                st.info("No pipelines configured")
                
                # Create new pipeline section
                st.subheader("➕ Create New Pipeline")
                with st.form("new_pipeline"):
                    pipeline_name = st.text_input("Pipeline Name")
                    pipeline_description = st.text_area("Description")
                    agent_id = st.text_input("Agent ID")
                    agent_type = st.selectbox("Agent Type", ["neural_network", "random_forest", "svm", "custom"])
                    
                    submitted = st.form_submit_button("Create Pipeline")
                    if submitted and pipeline_name:
                        st.success(f"Pipeline creation for '{pipeline_name}' would be implemented here")
                
        except Exception as e:
            st.error(f"Pipeline management error: {e}")
    
    def _render_model_registry(self):
        """Render model registry page"""
        st.title("📦 Model Registry")
        
        # This would integrate with MLflow Model Registry
        # For now, show placeholder content
        
        st.info("🚧 Model Registry integration coming soon!")
        
        # Mock model registry data
        models = [
            {
                'name': 'sutazai_sentiment_analyzer',
                'version': '1.2.0',
                'stage': 'Production',
                'description': 'Neural network for sentiment analysis',
                'created': '2025-01-15 10:30:00'
            },
            {
                'name': 'sutazai_image_classifier',
                'version': '2.1.0',
                'stage': 'Staging',
                'description': 'CNN for image classification',
                'created': '2025-01-10 14:20:00'
            }
        ]
        
        st.subheader("📋 Registered Models")
        
        for model in models:
            with st.expander(f"📦 {model['name']} v{model['version']} ({model['stage']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Version:** {model['version']}")
                    st.write(f"**Stage:** {model['stage']}")
                    st.write(f"**Created:** {model['created']}")
                
                with col2:
                    st.write(f"**Description:** {model['description']}")
                
                # Model actions
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button(f"🚀 Deploy", key=f"deploy_{model['name']}"):
                        st.success(f"Deployment initiated for {model['name']}")
                
                with col2:
                    if st.button(f"📊 Metrics", key=f"metrics_{model['name']}"):
                        st.info("Model metrics view would open here")
                
                with col3:
                    if st.button(f"📈 Monitor", key=f"monitor_{model['name']}"):
                        st.info("Model monitoring dashboard would open here")
                
                with col4:
                    if st.button(f"🔄 Version", key=f"version_{model['name']}"):
                        st.info("New version creation would start here")
    
    def _render_analysis(self):
        """Render analysis and comparison page"""
        st.title("📊 Analysis & Comparison")
        
        # Experiment comparison section
        st.subheader("🔬 Experiment Comparison")
        
        try:
            # Get available experiments
            experiments = self.client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            experiment_options = {exp.name: exp.experiment_id for exp in experiments}
            
            if experiment_options:
                # Select experiments to compare
                selected_experiments = st.multiselect(
                    "Select experiments to compare (2-5 recommended):",
                    options=list(experiment_options.keys()),
                    max_selections=5
                )
                
                if len(selected_experiments) >= 2:
                    # Metrics to compare
                    metrics_input = st.text_input(
                        "Metrics to compare (comma-separated):",
                        value="accuracy,loss,f1_score",
                        help="Leave empty to compare all available metrics"
                    )
                    
                    metrics_list = [m.strip() for m in metrics_input.split(',')] if metrics_input else None
                    
                    if st.button("🔍 Run Comparison"):
                        with st.spinner("Running experiment comparison..."):
                            try:
                                # Get experiment IDs
                                exp_ids = [experiment_options[name] for name in selected_experiments]
                                
                                # Run comparison
                                comparison_result = experiment_analyzer.compare_experiments(exp_ids, metrics_list)
                                
                                # Display results
                                st.success("Comparison completed!")
                                
                                # Metrics summary
                                st.subheader("📈 Comparison Summary")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Runs Analyzed", comparison_result.total_runs)
                                with col2:
                                    st.metric("Experiments Compared", len(comparison_result.experiment_ids))
                                
                                # Statistical tests
                                if comparison_result.statistical_tests:
                                    st.subheader("📊 Statistical Analysis")
                                    for metric, tests in comparison_result.statistical_tests.items():
                                        st.write(f"**{metric}:**")
                                        if 'anova' in tests:
                                            anova = tests['anova']
                                            significance = "Significant" if anova['significant'] else "Not significant"
                                            st.write(f"  • ANOVA: {significance} (p={anova['p_value']:.4f})")
                                
                                # Recommendations
                                st.subheader("💡 Recommendations")
                                for rec in comparison_result.recommendations:
                                    st.write(f"• {rec}")
                                
                                # Charts
                                if comparison_result.charts:
                                    st.subheader("📊 Visualizations")
                                    for chart_type, chart_path in comparison_result.charts.items():
                                        if os.path.exists(chart_path):
                                            st.image(chart_path, caption=chart_type.replace('_', ' ').title())
                                
                            except Exception as e:
                                st.error(f"Comparison failed: {e}")
                else:
                    st.info("Select at least 2 experiments to run comparison")
            else:
                st.info("No experiments available for comparison")
        
        except Exception as e:
            st.error(f"Analysis error: {e}")
        
        st.markdown("---")
        
        # Model performance analysis section
        st.subheader("🎯 Model Performance Analysis")
        
        model_name = st.text_input("Model Name", value="sutazai_model")
        time_window = st.slider("Analysis Time Window (days)", min_value=1, max_value=90, value=30)
        
        if st.button("📊 Analyze Model Performance"):
            with st.spinner("Analyzing model performance..."):
                try:
                    # This would run actual model performance analysis
                    st.info("Model performance analysis would run here with real data")
                    
                    # Mock results for demonstration
                    st.success("Analysis completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Accuracy", "0.856", delta="0.02")
                    with col2:
                        st.metric("Drift Score", "0.12", delta="-0.03")
                    with col3:
                        st.metric("Anomalies Detected", "2", delta="1")
                    
                    st.write("**Recommendations:**")
                    st.write("• Model performance is stable")
                    st.write("• Monitor for potential drift in next 7 days")
                    st.write("• Consider retraining if drift score exceeds 0.2")
                
                except Exception as e:
                    st.error(f"Performance analysis failed: {e}")
    
    def _render_system_health(self):
        """Render system health monitoring page"""
        st.title("🏥 System Health")
        
        try:
            # Get system metrics (mock data for now)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MLflow Server", "🟢 Online", delta="99.9% uptime")
            
            with col2:
                st.metric("Database", "🟢 Healthy", delta="<10ms latency")
            
            with col3:
                st.metric("Disk Usage", "45%", delta="5%")
            
            with col4:
                st.metric("Active Connections", "23", delta="3")
            
            # System resource usage chart
            st.subheader("📊 Resource Usage")
            
            # Generate mock time series data
            times = pd.date_range(start='2025-01-01', periods=24, freq='H')
            cpu_usage = np.random.normal(60, 10, 24)
            memory_usage = np.random.normal(45, 8, 24)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
                shared_xaxes=True
            )
            
            fig.add_trace(
                go.Scatter(x=times, y=cpu_usage, name='CPU', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=times, y=memory_usage, name='Memory', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_layout(height=400, title_text="System Resource Usage (Last 24 Hours)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent errors/alerts
            st.subheader("🚨 Recent Alerts")
            
            alerts = [
                {"Time": "2025-01-15 14:30", "Level": "Warning", "Message": "High CPU usage detected (85%)"},
                {"Time": "2025-01-15 12:15", "Level": "Info", "Message": "Experiment cleanup completed"},
                {"Time": "2025-01-15 10:45", "Level": "Warning", "Message": "Agent tracking timeout for agent-42"}
            ]
            
            alerts_df = pd.DataFrame(alerts)
            st.dataframe(alerts_df, use_container_width=True)
            
            # System configuration
            st.subheader("⚙️ System Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.write("**MLflow Configuration:**")
                st.code(f"""
Tracking URI: {mlflow_config.tracking_uri}
Artifact Root: {mlflow_config.artifact_root}
Max Concurrent Experiments: {mlflow_config.max_concurrent_experiments}
Batch Logging Size: {mlflow_config.batch_logging_size}
                """)
            
            with config_col2:
                st.write("**Database Configuration:**")
                st.code(f"""
Pool Size: {mlflow_config.db_pool_size}
Max Overflow: {mlflow_config.db_max_overflow}
Pool Timeout: {mlflow_config.db_pool_timeout}s
Pool Recycle: {mlflow_config.db_pool_recycle}s
                """)
            
        except Exception as e:
            st.error(f"System health error: {e}")
    
def run_dashboard():
    """Main function to run the Streamlit dashboard"""
    dashboard = MLflowDashboard()
    dashboard.run()


if __name__ == "__main__":
    run_dashboard()