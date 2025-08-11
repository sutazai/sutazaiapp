#!/usr/bin/env python3
"""
Multi-Modal Fusion Visualization and Debugging Tools

This module provides comprehensive visualization and debugging capabilities 
for the SutazAI multi-modal fusion system, including:

- Real-time fusion pipeline monitoring
- Cross-modal attention visualization
- Representation space analysis
- Performance metrics dashboards
- Error analysis and debugging tools

Integration Features:
- Web-based dashboard using Streamlit
- Interactive plots and charts
- Real-time WebSocket updates
- Export capabilities for analysis
- Integration with SutazAI monitoring infrastructure

Author: SutazAI Multi-Modal Fusion System
Version: 1.0.0
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import time
import asyncio
import websockets
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
import io
import base64

from ..core.multi_modal_fusion_coordinator import ModalityType, FusionStrategy
from ..core.unified_representation import UnifiedRepresentation, RepresentationLevel
from ..pipeline.realtime_fusion_pipeline import PipelineMetrics, ProcessingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization tools"""
    update_interval: float = 1.0
    max_data_points: int = 1000
    enable_real_time: bool = True
    websocket_url: str = "ws://localhost:8765"
    database_path: str = "/opt/sutazaiapp/fusion/data/visualization.db"
    export_directory: str = "/opt/sutazaiapp/fusion/exports"
    color_scheme: str = "plotly"
    show_debug_info: bool = True

class DataCollector:
    """Collects and stores data for visualization"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.metrics_buffer = deque(maxlen=config.max_data_points)
        self.fusion_results_buffer = deque(maxlen=config.max_data_points)
        self.representation_buffer = deque(maxlen=config.max_data_points)
        self.error_buffer = deque(maxlen=config.max_data_points)
        
        # Database setup
        self.db_path = Path(config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_database()
        
        # Real-time data collection
        self.websocket_thread = None
        self.is_collecting = False
        
    def _setup_database(self):
        """Setup SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                requests_processed INTEGER,
                requests_failed INTEGER,
                average_latency REAL,
                throughput_per_second REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                active_workers INTEGER,
                queue_sizes TEXT
            )
        ''')
        
        # Fusion results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fusion_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                timestamp REAL,
                fusion_strategy TEXT,
                modality_types TEXT,
                processing_latency REAL,
                confidence_scores TEXT,
                error TEXT
            )
        ''')
        
        # Representations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS representations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                representation_id TEXT,
                timestamp REAL,
                embedding_data TEXT,
                semantic_features TEXT,
                confidence_score REAL,
                modality_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_collection(self):
        """Start real-time data collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        if self.config.enable_real_time:
            self.websocket_thread = threading.Thread(target=self._websocket_loop)
            self.websocket_thread.daemon = True
            self.websocket_thread.start()
            logger.info("Started real-time data collection")
    
    def stop_collection(self):
        """Stop real-time data collection"""
        self.is_collecting = False
        if self.websocket_thread:
            self.websocket_thread.join(timeout=5.0)
            logger.info("Stopped real-time data collection")
    
    def _websocket_loop(self):
        """WebSocket data collection loop"""
        while self.is_collecting:
            try:
                asyncio.run(self._websocket_client())
            except Exception as e:
                logger.error(f"WebSocket collection error: {e}")
                time.sleep(5.0)  # Wait before reconnecting
    
    async def _websocket_client(self):
        """WebSocket client for real-time data"""
        try:
            async with websockets.connect(self.config.websocket_url) as websocket:
                logger.info(f"Connected to WebSocket: {self.config.websocket_url}")
                
                while self.is_collecting:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        self._process_websocket_data(data)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
    
    def _process_websocket_data(self, data: Dict[str, Any]):
        """Process incoming WebSocket data"""
        # Store metrics data
        if 'requests_processed' in data:
            self.metrics_buffer.append({
                'timestamp': time.time(),
                **data
            })
            
            # Store in database
            self._store_metrics(data)
    
    def _store_metrics(self, data: Dict[str, Any]):
        """Store metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (
                    timestamp, requests_processed, requests_failed,
                    average_latency, throughput_per_second, memory_usage_mb,
                    cpu_usage_percent, active_workers, queue_sizes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                data.get('requests_processed', 0),
                data.get('requests_failed', 0),
                data.get('average_latency', 0.0),
                data.get('throughput_per_second', 0.0),
                data.get('memory_usage_mb', 0.0),
                data.get('cpu_usage_percent', 0.0),
                data.get('active_workers', 0),
                json.dumps(data.get('queue_sizes', {}))
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database storage error: {e}")
    
    def get_metrics_data(self, hours: int = 24) -> pd.DataFrame:
        """Retrieve metrics data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = time.time() - (hours * 3600)
        query = '''
            SELECT * FROM metrics 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        # Convert timestamp to datetime
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def add_fusion_result(self, result: ProcessingResponse):
        """Add fusion result to buffer and database"""
        result_data = {
            'request_id': result.request_id,
            'timestamp': result.timestamp,
            'processing_latency': result.processing_latency,
            'error': result.error,
            'fusion_strategy': getattr(result.fusion_result, 'fusion_strategy', 'unknown'),
            'modality_types': getattr(result.fusion_result, 'contributing_modalities', []),
            'confidence_scores': getattr(result.fusion_result, 'confidence_scores', {})
        }
        
        self.fusion_results_buffer.append(result_data)
        
        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fusion_results (
                    request_id, timestamp, fusion_strategy, modality_types,
                    processing_latency, confidence_scores, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_data['request_id'],
                result_data['timestamp'],
                str(result_data['fusion_strategy']),
                json.dumps([str(m) for m in result_data['modality_types']]),
                result_data['processing_latency'],
                json.dumps(result_data['confidence_scores']),
                result_data['error']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Fusion result storage error: {e}")
    
    def add_representation(self, representation: UnifiedRepresentation):
        """Add representation to buffer and database"""
        repr_data = {
            'representation_id': representation.representation_id,
            'timestamp': representation.timestamp,
            'embedding_data': representation.unified_embedding.tolist(),
            'semantic_features': representation.semantic_features,
            'confidence_score': representation.confidence_score,
            'modality_count': len(representation.modality_embeddings)
        }
        
        self.representation_buffer.append(repr_data)
        
        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO representations (
                    representation_id, timestamp, embedding_data,
                    semantic_features, confidence_score, modality_count
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                repr_data['representation_id'],
                repr_data['timestamp'],
                json.dumps(repr_data['embedding_data']),
                json.dumps(repr_data['semantic_features']),
                repr_data['confidence_score'],
                repr_data['modality_count']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Representation storage error: {e}")

class FusionVisualizer:
    """Main visualization interface"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.data_collector = DataCollector(self.config)
        
        # Streamlit configuration
        st.set_page_config(
            page_title="SutazAI Multi-Modal Fusion Dashboard",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._inject_custom_css()
        
    def _inject_custom_css(self):
        """Inject custom CSS for better styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .status-healthy {
            color: #28a745;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Run the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🔬 SutazAI Multi-Modal Fusion Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Real-Time Metrics", 
            "🔄 Fusion Analysis", 
            "🧠 Representation Space",
            "🔍 Debugging Tools",
            "📈 Performance Analysis"
        ])
        
        with tab1:
            self._render_realtime_metrics()
        
        with tab2:
            self._render_fusion_analysis()
        
        with tab3:
            self._render_representation_space()
        
        with tab4:
            self._render_debugging_tools()
        
        with tab5:
            self._render_performance_analysis()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Controls")
        
        # Data collection controls
        st.sidebar.subheader("Data Collection")
        
        if st.sidebar.button("🚀 Start Collection"):
            self.data_collector.start_collection()
            st.sidebar.success("Started data collection")
        
        if st.sidebar.button("⏹️ Stop Collection"):
            self.data_collector.stop_collection()
            st.sidebar.info("Stopped data collection")
        
        # Time range selection
        st.sidebar.subheader("Time Range")
        time_range = st.sidebar.selectbox(
            "Select time range:",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
            index=2
        )
        
        hours_map = {
            "Last Hour": 1,
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last Week": 168
        }
        self.selected_hours = hours_map[time_range]
        
        # Display settings
        st.sidebar.subheader("Display Settings")
        self.show_debug = st.sidebar.checkbox("Show Debug Info", value=self.config.show_debug_info)
        self.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        
        if self.auto_refresh:
            self.refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 30, 5)
        
        # Export options
        st.sidebar.subheader("Export")
        if st.sidebar.button("📊 Export Data"):
            self._export_data()
        
        if st.sidebar.button("📈 Export Plots"):
            self._export_plots()
    
    def _render_realtime_metrics(self):
        """Render real-time metrics tab"""
        st.subheader("📊 Real-Time Pipeline Metrics")
        
        # Get metrics data
        metrics_df = self.data_collector.get_metrics_data(self.selected_hours)
        
        if metrics_df.empty:
            st.warning("No metrics data available. Start data collection to see real-time metrics.")
            return
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        latest_metrics = metrics_df.iloc[0] if not metrics_df.empty else {}
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Throughput",
                f"{latest_metrics.get('throughput_per_second', 0):.1f} req/s",
                delta=f"{latest_metrics.get('throughput_per_second', 0) - 100:.1f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Avg Latency",
                f"{latest_metrics.get('average_latency', 0):.3f}s",
                delta=f"{latest_metrics.get('average_latency', 0) - 0.1:.3f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Memory Usage",
                f"{latest_metrics.get('memory_usage_mb', 0):.1f} MB",
                delta=f"{latest_metrics.get('memory_usage_mb', 0) - 1000:.1f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Active Workers",
                f"{latest_metrics.get('active_workers', 0)}",
                delta=f"{latest_metrics.get('active_workers', 0) - 4}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Time series plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Throughput and latency over time
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Throughput (req/s)', 'Average Latency (s)'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['datetime'],
                    y=metrics_df['throughput_per_second'],
                    mode='lines+markers',
                    name='Throughput',
                    line=dict(color='#1f77b4')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['datetime'],
                    y=metrics_df['average_latency'],
                    mode='lines+markers',
                    name='Latency',
                    line=dict(color='#ff7f0e')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=400, title_text="Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Resource usage
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Memory Usage (MB)', 'CPU Usage (%)'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['datetime'],
                    y=metrics_df['memory_usage_mb'],
                    mode='lines+markers',
                    name='Memory',
                    line=dict(color='#2ca02c')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['datetime'],
                    y=metrics_df['cpu_usage_percent'],
                    mode='lines+markers',
                    name='CPU',
                    line=dict(color='#d62728')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=400, title_text="Resource Usage")
            st.plotly_chart(fig, use_container_width=True)
        
        # Queue sizes (if available)
        if 'queue_sizes' in metrics_df.columns:
            st.subheader("Queue Status")
            
            # Parse queue sizes from JSON strings
            queue_data = []
            for _, row in metrics_df.iterrows():
                try:
                    queue_sizes = json.loads(row['queue_sizes'])
                    for queue_name, size in queue_sizes.items():
                        queue_data.append({
                            'datetime': row['datetime'],
                            'queue': queue_name,
                            'size': size
                        })
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.debug(f"Continuing after exception: {e}")
                    continue
            
            if queue_data:
                queue_df = pd.DataFrame(queue_data)
                
                fig = px.line(
                    queue_df, 
                    x='datetime', 
                    y='size', 
                    color='queue',
                    title="Queue Sizes Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_fusion_analysis(self):
        """Render fusion analysis tab"""
        st.subheader("🔄 Fusion Strategy Analysis")
        
        # Fusion results summary
        fusion_data = list(self.data_collector.fusion_results_buffer)
        
        if not fusion_data:
            st.warning("No fusion results available. Process some requests to see analysis.")
            return
        
        df = pd.DataFrame(fusion_data)
        
        # Strategy distribution
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_counts = df['fusion_strategy'].value_counts()
            fig = px.pie(
                values=strategy_counts.values,
                names=strategy_counts.index,
                title="Fusion Strategy Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by strategy
            success_df = df.copy()
            success_df['success'] = success_df['error'].isna()
            success_rates = success_df.groupby('fusion_strategy')['success'].mean()
            
            fig = px.bar(
                x=success_rates.index,
                y=success_rates.values,
                title="Success Rate by Strategy",
                labels={'y': 'Success Rate', 'x': 'Fusion Strategy'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Latency analysis
        st.subheader("Latency Analysis")
        
        fig = px.box(
            df,
            x='fusion_strategy',
            y='processing_latency',
            title="Processing Latency by Fusion Strategy"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence scores analysis
        if 'confidence_scores' in df.columns:
            st.subheader("Confidence Score Analysis")
            
            # Extract confidence scores
            confidence_data = []
            for _, row in df.iterrows():
                try:
                    scores = json.loads(row['confidence_scores']) if isinstance(row['confidence_scores'], str) else row['confidence_scores']
                    for modality, score in scores.items():
                        confidence_data.append({
                            'request_id': row['request_id'],
                            'modality': modality,
                            'confidence': score,
                            'strategy': row['fusion_strategy']
                        })
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.debug(f"Continuing after exception: {e}")
                    continue
            
            if confidence_data:
                conf_df = pd.DataFrame(confidence_data)
                
                fig = px.violin(
                    conf_df,
                    x='modality',
                    y='confidence',
                    color='strategy',
                    title="Confidence Scores by Modality and Strategy"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_representation_space(self):
        """Render representation space visualization"""
        st.subheader("🧠 Unified Representation Space")
        
        # Get representation data
        repr_data = list(self.data_collector.representation_buffer)
        
        if not repr_data:
            st.warning("No representation data available.")
            return
        
        # Extract embeddings for dimensionality reduction
        embeddings = []
        metadata = []
        
        for repr_item in repr_data[-100:]:  # Last 100 representations
            try:
                embedding = np.array(repr_item['embedding_data'])
                embeddings.append(embedding)
                metadata.append({
                    'id': repr_item['representation_id'],
                    'confidence': repr_item['confidence_score'],
                    'modality_count': repr_item['modality_count'],
                    'semantic_features': repr_item['semantic_features']
                })
            except (IOError, OSError, FileNotFoundError) as e:
                logger.debug(f"Continuing after exception: {e}")
                continue
        
        if len(embeddings) < 2:
            st.warning("Need at least 2 representations for visualization.")
            return
        
        embeddings = np.array(embeddings)
        
        # Dimensionality reduction options
        reduction_method = st.selectbox(
            "Select dimensionality reduction method:",
            ["t-SNE", "PCA", "UMAP"]
        )
        
        # Apply dimensionality reduction
        if reduction_method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
        elif reduction_method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
        else:  # UMAP
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings)
            except ImportError:
                st.error("UMAP not available. Please install umap-learn.")
                return
        
        # Create visualization DataFrame
        viz_df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'confidence': [m['confidence'] for m in metadata],
            'modality_count': [m['modality_count'] for m in metadata],
            'id': [m['id'] for m in metadata]
        })
        
        # Interactive scatter plot
        fig = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='confidence',
            size='modality_count',
            hover_data=['id'],
            title=f"Representation Space ({reduction_method})",
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Semantic feature analysis
        st.subheader("Semantic Feature Distribution")
        
        # Extract semantic features
        semantic_data = []
        for item in metadata:
            try:
                features = item['semantic_features']
                for feature_name, feature_value in features.items():
                    if isinstance(feature_value, (int, float)):
                        semantic_data.append({
                            'feature': feature_name,
                            'value': feature_value,
                            'representation_id': item['id']
                        })
            except (IOError, OSError, FileNotFoundError) as e:
                logger.debug(f"Continuing after exception: {e}")
                continue
        
        if semantic_data:
            semantic_df = pd.DataFrame(semantic_data)
            
            fig = px.box(
                semantic_df,
                x='feature',
                y='value',
                title="Semantic Feature Value Distribution"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_debugging_tools(self):
        """Render debugging tools tab"""
        st.subheader("🔍 Debugging Tools")
        
        # Error analysis
        st.subheader("Error Analysis")
        
        fusion_data = list(self.data_collector.fusion_results_buffer)
        if fusion_data:
            df = pd.DataFrame(fusion_data)
            errors_df = df[df['error'].notna()]
            
            if not errors_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Error frequency
                    error_counts = errors_df['error'].value_counts()
                    fig = px.bar(
                        x=error_counts.values,
                        y=error_counts.index,
                        orientation='h',
                        title="Most Common Errors"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Error rate over time
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                    df['has_error'] = df['error'].notna()
                    
                    # Group by hour and calculate error rate
                    hourly_errors = df.set_index('datetime').resample('1H')['has_error'].mean()
                    
                    fig = px.line(
                        x=hourly_errors.index,
                        y=hourly_errors.values,
                        title="Error Rate Over Time",
                        labels={'y': 'Error Rate', 'x': 'Time'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Error details table
                st.subheader("Recent Errors")
                error_table = errors_df[['request_id', 'timestamp', 'fusion_strategy', 'error']].head(20)
                st.dataframe(error_table)
            else:
                st.success("No errors found in recent data!")
        
        # System health indicators
        st.subheader("System Health Indicators")
        
        metrics_df = self.data_collector.get_metrics_data(1)  # Last hour
        
        if not metrics_df.empty:
            latest = metrics_df.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Memory health
                memory_usage = latest.get('memory_usage_mb', 0)
                memory_status = "healthy" if memory_usage < 2000 else "warning" if memory_usage < 4000 else "error"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Memory Usage</h4>
                    <p class="status-{memory_status}">{memory_usage:.1f} MB ({memory_status.upper()})</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # CPU health
                cpu_usage = latest.get('cpu_usage_percent', 0)
                cpu_status = "healthy" if cpu_usage < 70 else "warning" if cpu_usage < 90 else "error"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>CPU Usage</h4>
                    <p class="status-{cpu_status}">{cpu_usage:.1f}% ({cpu_status.upper()})</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Latency health
                latency = latest.get('average_latency', 0)
                latency_status = "healthy" if latency < 0.1 else "warning" if latency < 1.0 else "error"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Average Latency</h4>
                    <p class="status-{latency_status}">{latency:.3f}s ({latency_status.upper()})</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Debug information display
        if self.show_debug:
            st.subheader("Debug Information")
            
            debug_info = {
                'collector_status': 'running' if self.data_collector.is_collecting else 'stopped',
                'metrics_buffer_size': len(self.data_collector.metrics_buffer),
                'fusion_buffer_size': len(self.data_collector.fusion_results_buffer),
                'representation_buffer_size': len(self.data_collector.representation_buffer),
                'database_path': str(self.data_collector.db_path),
                'websocket_url': self.config.websocket_url
            }
            
            st.json(debug_info)
    
    def _render_performance_analysis(self):
        """Render performance analysis tab"""
        st.subheader("📈 Performance Analysis")
        
        metrics_df = self.data_collector.get_metrics_data(self.selected_hours)
        
        if metrics_df.empty:
            st.warning("No performance data available.")
            return
        
        # Performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_throughput = metrics_df['throughput_per_second'].mean()
            st.metric("Avg Throughput", f"{avg_throughput:.1f} req/s")
        
        with col2:
            avg_latency = metrics_df['average_latency'].mean()
            st.metric("Avg Latency", f"{avg_latency:.3f}s")
        
        with col3:
            max_memory = metrics_df['memory_usage_mb'].max()
            st.metric("Peak Memory", f"{max_memory:.1f} MB")
        
        with col4:
            avg_workers = metrics_df['active_workers'].mean()
            st.metric("Avg Workers", f"{avg_workers:.1f}")
        
        # Performance trends
        st.subheader("Performance Trends")
        
        # Create performance score
        # Normalize metrics and create composite score
        normalized_throughput = (metrics_df['throughput_per_second'] - metrics_df['throughput_per_second'].min()) / (metrics_df['throughput_per_second'].max() - metrics_df['throughput_per_second'].min() + 1e-8)
        normalized_latency = 1 - ((metrics_df['average_latency'] - metrics_df['average_latency'].min()) / (metrics_df['average_latency'].max() - metrics_df['average_latency'].min() + 1e-8))
        normalized_memory = 1 - ((metrics_df['memory_usage_mb'] - metrics_df['memory_usage_mb'].min()) / (metrics_df['memory_usage_mb'].max() - metrics_df['memory_usage_mb'].min() + 1e-8))
        
        performance_score = (normalized_throughput + normalized_latency + normalized_memory) / 3
        
        fig = px.line(
            x=metrics_df['datetime'],
            y=performance_score,
            title="Overall Performance Score (0-1, higher is better)",
            labels={'y': 'Performance Score', 'x': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Metric Correlations")
        
        numeric_cols = ['throughput_per_second', 'average_latency', 'memory_usage_mb', 'cpu_usage_percent', 'active_workers']
        correlation_matrix = metrics_df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Metric Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance recommendations
        st.subheader("Performance Recommendations")
        
        recommendations = []
        
        # Analyze current performance
        latest_metrics = metrics_df.iloc[0]
        
        if latest_metrics['average_latency'] > 1.0:
            recommendations.append("⚠️ High latency detected. Consider scaling up workers or optimizing fusion algorithms.")
        
        if latest_metrics['memory_usage_mb'] > 3000:
            recommendations.append("⚠️ High memory usage. Consider implementing more aggressive garbage collection.")
        
        if latest_metrics['cpu_usage_percent'] > 80:
            recommendations.append("⚠️ High CPU usage. Consider distributing load across more processes.")
        
        if latest_metrics['throughput_per_second'] < 10:
            recommendations.append("⚠️ Low throughput. Check for bottlenecks in the processing pipeline.")
        
        if not recommendations:
            recommendations.append("✅ System performance looks good!")
        
        for rec in recommendations:
            st.write(rec)
    
    def _export_data(self):
        """Export data to files"""
        export_dir = Path(self.config.export_directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export metrics
        metrics_df = self.data_collector.get_metrics_data(self.selected_hours)
        if not metrics_df.empty:
            metrics_file = export_dir / f"metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            st.success(f"Metrics exported to {metrics_file}")
        
        # Export fusion results
        fusion_data = list(self.data_collector.fusion_results_buffer)
        if fusion_data:
            fusion_df = pd.DataFrame(fusion_data)
            fusion_file = export_dir / f"fusion_results_{timestamp}.csv"
            fusion_df.to_csv(fusion_file, index=False)
            st.success(f"Fusion results exported to {fusion_file}")
    
    def _export_plots(self):
        """Export plots as images"""
        export_dir = Path(self.config.export_directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.info(f"Plots would be exported to {export_dir} (implementation pending)")

# Streamlit app entry point
def main():
    """Main entry point for Streamlit app"""
    config = VisualizationConfig()
    visualizer = FusionVisualizer(config)
    visualizer.run_dashboard()

if __name__ == "__main__":
    main()