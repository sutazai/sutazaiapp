"""
Real-time Performance Monitoring Dashboard for SutazAI System
Provides live performance metrics, alerts, and AI-powered insights
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import psutil
import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@dataclass 
class MetricSnapshot:
    """Single metric snapshot"""
    timestamp: float
    component: str
    metric_name: str
    value: float
    unit: str
    status: str  # normal, warning, critical
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    id: str
    severity: str  # low, medium, high, critical
    component: str
    metric: str
    message: str
    threshold: float
    current_value: float
    timestamp: float
    resolved: bool = False
    
class AIPerformanceAnalyzer:
    """AI-powered performance analysis and prediction"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.metric_history = {}
        self.alert_predictor = None
        self.model_initialized = False
        
    async def initialize(self):
        """Initialize AI models"""
        try:
            # Initialize for basic text analysis if needed
            self.model_initialized = True
            logger.info("AI Performance Analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            
    def analyze_metrics(self, metrics: List[MetricSnapshot]) -> Dict[str, Any]:
        """Analyze metrics for patterns and anomalies"""
        if not metrics:
            return {"status": "no_data", "insights": []}
            
        # Group metrics by component and metric name
        grouped_metrics = {}
        for metric in metrics:
            key = f"{metric.component}_{metric.metric_name}"
            if key not in grouped_metrics:
                grouped_metrics[key] = []
            grouped_metrics[key].append(metric)
            
        insights = []
        anomalies = []
        
        for key, metric_list in grouped_metrics.items():
            if len(metric_list) < 10:  # Need minimum data points
                continue
                
            values = [m.value for m in metric_list]
            timestamps = [m.timestamp for m in metric_list]
            
            # Statistical analysis
            mean_val = np.mean(values)
            std_val = np.std(values)
            trend = self._calculate_trend(values)
            
            # Anomaly detection
            if len(values) >= 10:
                values_reshaped = np.array(values).reshape(-1, 1)
                anomaly_scores = self.anomaly_detector.fit_predict(values_reshaped)
                anomaly_count = sum(1 for score in anomaly_scores if score == -1)
                
                if anomaly_count > 0:
                    anomalies.append({
                        "metric": key,
                        "anomaly_count": anomaly_count,
                        "anomaly_percentage": anomaly_count / len(values) * 100,
                        "latest_value": values[-1],
                        "mean_value": mean_val
                    })
            
            # Generate insights
            if trend == "increasing" and "cpu" in key.lower():
                insights.append(f"CPU usage trending upward for {key} - consider scaling")
            elif trend == "increasing" and "memory" in key.lower():
                insights.append(f"Memory usage trending upward for {key} - potential memory leak")
            elif trend == "increasing" and "response_time" in key.lower():
                insights.append(f"Response times increasing for {key} - performance degradation detected")
                
        return {
            "status": "analyzed",
            "insights": insights,
            "anomalies": anomalies,
            "total_metrics": len(metrics),
            "analysis_timestamp": time.time()
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return "insufficient_data"
            
        # Simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        threshold = np.std(values) * 0.1  # 10% of standard deviation
        
        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        else:
            return "stable"
            
    def predict_future_performance(self, metrics: List[MetricSnapshot], horizon_minutes: int = 60) -> Dict[str, Any]:
        """Predict future performance based on current trends"""
        predictions = {}
        current_time = time.time()
        future_time = current_time + (horizon_minutes * 60)
        
        # Group metrics by type
        grouped_metrics = {}
        for metric in metrics:
            key = f"{metric.component}_{metric.metric_name}"
            if key not in grouped_metrics:
                grouped_metrics[key] = []
            grouped_metrics[key].append(metric)
            
        for key, metric_list in grouped_metrics.items():
            if len(metric_list) < 10:
                continue
                
            values = [m.value for m in metric_list]
            timestamps = [m.timestamp for m in metric_list]
            
            # Simple linear extrapolation
            try:
                coeffs = np.polyfit(timestamps, values, 1)
                predicted_value = coeffs[0] * future_time + coeffs[1]
                
                current_value = values[-1]
                change_percentage = ((predicted_value - current_value) / current_value) * 100 if current_value != 0 else 0
                
                predictions[key] = {
                    "current_value": current_value,
                    "predicted_value": max(0, predicted_value),  # Ensure non-negative
                    "change_percentage": change_percentage,
                    "confidence": min(100, len(metric_list) * 5),  # Confidence based on data points
                    "trend": "increasing" if coeffs[0] > 0 else "decreasing" if coeffs[0] < 0 else "stable"
                }
                
            except Exception as e:
                logger.debug(f"Failed to predict for {key}: {e}")
                continue
                
        return {
            "predictions": predictions,
            "horizon_minutes": horizon_minutes,
            "prediction_timestamp": current_time
        }

class PerformanceMonitor:
    """Core performance monitoring system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.redis_client: Optional[redis.Redis] = None
        self.ai_analyzer = AIPerformanceAnalyzer()
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.metric_history: List[MetricSnapshot] = []
        self.max_history_size = 10000
        self.connected_clients: List[WebSocket] = []
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70, "critical": 85},
            "memory_usage": {"warning": 75, "critical": 90},
            "disk_usage": {"warning": 80, "critical": 95},
            "response_time": {"warning": 1000, "critical": 5000},  # milliseconds
            "error_rate": {"warning": 5, "critical": 10},  # percentage
            "throughput": {"warning": 50, "critical": 10}  # req/s minimum
        }
        
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            # Initialize Redis connection
            self.redis_client = await redis.from_url("redis://localhost:6379/0", decode_responses=True)
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            
            # Initialize AI analyzer
            await self.ai_analyzer.initialize()
            
            logger.info("Performance monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitor: {e}")
            raise
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
            
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring"""
        logger.info(f"Starting performance monitoring with {interval_seconds}s interval")
        
        while True:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                
                # Combine all metrics
                all_metrics = system_metrics + app_metrics
                
                # Add to history
                self.metric_history.extend(all_metrics)
                
                # Trim history if too large
                if len(self.metric_history) > self.max_history_size:
                    self.metric_history = self.metric_history[-self.max_history_size:]
                    
                # Check for alerts
                new_alerts = self._check_thresholds(all_metrics)
                for alert in new_alerts:
                    self.active_alerts[alert.id] = alert
                    
                # Store metrics in Redis
                await self._store_metrics(all_metrics)
                
                # Broadcast to connected clients
                await self._broadcast_metrics(all_metrics)
                
                # Run AI analysis periodically (every 5 minutes)
                if len(self.metric_history) % 10 == 0:  # Every 10 cycles
                    ai_insights = self.ai_analyzer.analyze_metrics(self.metric_history[-500:])
                    await self._broadcast_ai_insights(ai_insights)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            await asyncio.sleep(interval_seconds)
            
    async def _collect_system_metrics(self) -> List[MetricSnapshot]:
        """Collect system-level performance metrics"""
        current_time = time.time()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        metrics.append(MetricSnapshot(
            timestamp=current_time,
            component="system",
            metric_name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            status=self._get_status("cpu_usage", cpu_percent),
            metadata={"cpu_count": cpu_count}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(MetricSnapshot(
            timestamp=current_time,
            component="system",
            metric_name="memory_usage",
            value=memory.percent,
            unit="percent",
            status=self._get_status("memory_usage", memory.percent),
            metadata={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3)
            }
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(MetricSnapshot(
            timestamp=current_time,
            component="system",
            metric_name="disk_usage",
            value=disk_percent,
            unit="percent",
            status=self._get_status("disk_usage", disk_percent),
            metadata={
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3)
            }
        ))
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics.extend([
            MetricSnapshot(
                timestamp=current_time,
                component="system",
                metric_name="network_bytes_sent",
                value=net_io.bytes_sent,
                unit="bytes",
                status="normal"
            ),
            MetricSnapshot(
                timestamp=current_time,
                component="system",
                metric_name="network_bytes_recv",
                value=net_io.bytes_recv,
                unit="bytes",
                status="normal"
            )
        ])
        
        # Process-specific metrics
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if proc_info['name'] in ['python', 'uvicorn', 'gunicorn', 'ollama']:
                    metrics.append(MetricSnapshot(
                        timestamp=current_time,
                        component=f"process_{proc_info['name']}",
                        metric_name="cpu_usage",
                        value=proc_info['cpu_percent'] or 0,
                        unit="percent",
                        status="normal",
                        metadata={"pid": proc_info['pid']}
                    ))
                    
                    metrics.append(MetricSnapshot(
                        timestamp=current_time,
                        component=f"process_{proc_info['name']}",
                        metric_name="memory_usage",
                        value=proc_info['memory_percent'] or 0,
                        unit="percent",
                        status="normal",
                        metadata={"pid": proc_info['pid']}
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return metrics
        
    async def _collect_application_metrics(self) -> List[MetricSnapshot]:
        """Collect application-specific metrics"""
        current_time = time.time()
        metrics = []
        
        if not self.session:
            return metrics
            
        # Test key endpoints for response time and availability
        endpoints = [
            "/health",
            "/api/v1/system/status",
            "/api/v1/agents/",
            "/api/v1/models/"
        ]
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    # Response time metric
                    metrics.append(MetricSnapshot(
                        timestamp=current_time,
                        component="api",
                        metric_name=f"response_time_{endpoint.replace('/', '_')}",
                        value=response_time,
                        unit="milliseconds",
                        status=self._get_status("response_time", response_time),
                        metadata={"endpoint": endpoint, "status_code": response.status}
                    ))
                    
                    # Availability metric (1 for success, 0 for failure)
                    availability = 1 if 200 <= response.status < 300 else 0
                    metrics.append(MetricSnapshot(
                        timestamp=current_time,
                        component="api",
                        metric_name=f"availability_{endpoint.replace('/', '_')}",
                        value=availability,
                        unit="boolean",
                        status="normal" if availability else "critical",
                        metadata={"endpoint": endpoint, "status_code": response.status}
                    ))
                    
            except Exception as e:
                logger.debug(f"Failed to collect metrics for {endpoint}: {e}")
                
                # Record as unavailable
                metrics.append(MetricSnapshot(
                    timestamp=current_time,
                    component="api",
                    metric_name=f"availability_{endpoint.replace('/', '_')}",
                    value=0,
                    unit="boolean",
                    status="critical",
                    metadata={"endpoint": endpoint, "error": str(e)}
                ))
                
        # Database connection test (if available)
        try:
            async with self.session.get(f"{self.base_url}/api/v1/system/status") as response:
                if response.status == 200:
                    data = await response.json()
                    if "database" in data:
                        db_status = data["database"]
                        metrics.append(MetricSnapshot(
                            timestamp=current_time,
                            component="database",
                            metric_name="connection_status",
                            value=1 if db_status == "connected" else 0,
                            unit="boolean",
                            status="normal" if db_status == "connected" else "critical"
                        ))
        except Exception as e:
            logger.debug(f"Failed to collect database metrics: {e}")
            
        return metrics
        
    def _get_status(self, metric_type: str, value: float) -> str:
        """Determine status based on threshold"""
        if metric_type not in self.thresholds:
            return "normal"
            
        thresholds = self.thresholds[metric_type]
        
        if value >= thresholds["critical"]:
            return "critical"
        elif value >= thresholds["warning"]:
            return "warning"
        else:
            return "normal"
            
    def _check_thresholds(self, metrics: List[MetricSnapshot]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts"""
        new_alerts = []
        current_time = time.time()
        
        for metric in metrics:
            if metric.status in ["warning", "critical"]:
                alert_id = f"{metric.component}_{metric.metric_name}_{metric.status}"
                
                # Check if this alert already exists and is recent
                if alert_id in self.active_alerts:
                    existing_alert = self.active_alerts[alert_id]
                    if current_time - existing_alert.timestamp < 300:  # 5 minutes
                        continue  # Don't duplicate recent alerts
                        
                # Create new alert
                alert = PerformanceAlert(
                    id=alert_id,
                    severity=metric.status,
                    component=metric.component,
                    metric=metric.metric_name,
                    message=f"{metric.component} {metric.metric_name} is {metric.status}: {metric.value}{metric.unit}",
                    threshold=self.thresholds.get(metric.metric_name, {}).get(metric.status, 0),
                    current_value=metric.value,
                    timestamp=current_time
                )
                
                new_alerts.append(alert)
                
        return new_alerts
        
    async def _store_metrics(self, metrics: List[MetricSnapshot]):
        """Store metrics in Redis for persistence"""
        if not self.redis_client:
            return
            
        try:
            pipe = self.redis_client.pipeline()
            
            for metric in metrics:
                key = f"metrics:{metric.component}:{metric.metric_name}"
                value = json.dumps(asdict(metric))
                
                # Store with expiration (24 hours)
                pipe.setex(key, 86400, value)
                
                # Also add to time series
                ts_key = f"timeseries:{metric.component}:{metric.metric_name}"
                pipe.zadd(ts_key, {value: metric.timestamp})
                
                # Keep only last 1000 points in time series
                pipe.zremrangebyrank(ts_key, 0, -1001)
                
            await pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {e}")
            
    async def _broadcast_metrics(self, metrics: List[MetricSnapshot]):
        """Broadcast metrics to connected WebSocket clients"""
        if not self.connected_clients:
            return
            
        try:
            message = {
                "type": "metrics_update",
                "timestamp": time.time(),
                "metrics": [asdict(metric) for metric in metrics]
            }
            
            # Send to all connected clients
            disconnected_clients = []
            for client in self.connected_clients:
                try:
                    await client.send_text(json.dumps(message))
                except Exception:
                    disconnected_clients.append(client)
                    
            # Remove disconnected clients
            for client in disconnected_clients:
                self.connected_clients.remove(client)
                
        except Exception as e:
            logger.error(f"Failed to broadcast metrics: {e}")
            
    async def _broadcast_ai_insights(self, insights: Dict[str, Any]):
        """Broadcast AI insights to connected clients"""
        if not self.connected_clients:
            return
            
        try:
            message = {
                "type": "ai_insights",
                "timestamp": time.time(),
                "insights": insights
            }
            
            disconnected_clients = []
            for client in self.connected_clients:
                try:
                    await client.send_text(json.dumps(message))
                except Exception:
                    disconnected_clients.append(client)
                    
            for client in disconnected_clients:
                self.connected_clients.remove(client)
                
        except Exception as e:
            logger.error(f"Failed to broadcast AI insights: {e}")
            
    async def add_websocket_client(self, websocket: WebSocket):
        """Add WebSocket client for real-time updates"""
        self.connected_clients.append(websocket)
        
        # Send current metrics to new client
        if self.metric_history:
            recent_metrics = self.metric_history[-50:]  # Last 50 metrics
            await websocket.send_text(json.dumps({
                "type": "initial_metrics",
                "metrics": [asdict(metric) for metric in recent_metrics]
            }))
            
    async def remove_websocket_client(self, websocket: WebSocket):
        """Remove WebSocket client"""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
            
    async def get_historical_metrics(self, component: str, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics for a specific component and metric"""
        if not self.redis_client:
            return []
            
        try:
            ts_key = f"timeseries:{component}:{metric_name}"
            start_time = time.time() - (hours * 3600)
            
            # Get time series data
            data = await self.redis_client.zrangebyscore(
                ts_key, start_time, time.time(), withscores=True
            )
            
            metrics = []
            for value_json, timestamp in data:
                try:
                    metric_data = json.loads(value_json)
                    metrics.append(metric_data)
                except json.JSONDecodeError:
                    continue
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            return []

# FastAPI integration
app = FastAPI(title="SutazAI Performance Monitoring Dashboard")

# Global monitor instance
monitor = PerformanceMonitor()

@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup"""
    await monitor.initialize()
    # Start monitoring in background
    asyncio.create_task(monitor.start_monitoring())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await monitor.cleanup()

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    await monitor.add_websocket_client(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": time.time()}))
    except WebSocketDisconnect:
        await monitor.remove_websocket_client(websocket)

@app.get("/api/metrics/current")
async def get_current_metrics():
    """Get current performance metrics"""
    if not monitor.metric_history:
        raise HTTPException(status_code=404, detail="No metrics available")
        
    # Get latest metrics (last 30 seconds)
    current_time = time.time()
    recent_metrics = [
        asdict(metric) for metric in monitor.metric_history
        if current_time - metric.timestamp <= 30
    ]
    
    return {
        "timestamp": current_time,
        "metrics": recent_metrics,
        "count": len(recent_metrics)
    }

@app.get("/api/metrics/historical/{component}/{metric_name}")
async def get_historical_metrics(component: str, metric_name: str, hours: int = 24):
    """Get historical metrics for specific component and metric"""
    metrics = await monitor.get_historical_metrics(component, metric_name, hours)
    
    return {
        "component": component,
        "metric_name": metric_name,
        "hours": hours,
        "metrics": metrics,
        "count": len(metrics)
    }

@app.get("/api/alerts/active")
async def get_active_alerts():
    """Get active performance alerts"""
    alerts = [asdict(alert) for alert in monitor.active_alerts.values() if not alert.resolved]
    
    return {
        "timestamp": time.time(),
        "alerts": alerts,
        "count": len(alerts)
    }

@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve a performance alert"""
    if alert_id not in monitor.active_alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    monitor.active_alerts[alert_id].resolved = True
    
    return {
        "alert_id": alert_id,
        "status": "resolved",
        "timestamp": time.time()
    }

@app.get("/api/analysis/ai-insights")
async def get_ai_insights():
    """Get AI-powered performance insights"""
    if not monitor.metric_history:
        raise HTTPException(status_code=404, detail="No metrics available for analysis")
        
    # Analyze last 1000 metrics or last hour, whichever is smaller
    current_time = time.time()
    recent_metrics = [
        metric for metric in monitor.metric_history
        if current_time - metric.timestamp <= 3600  # Last hour
    ][-1000:]  # Last 1000 points max
    
    insights = monitor.ai_analyzer.analyze_metrics(recent_metrics)
    
    return {
        "timestamp": current_time,
        "analysis": insights,
        "metrics_analyzed": len(recent_metrics)
    }

@app.get("/api/analysis/predictions")
async def get_performance_predictions(horizon_minutes: int = 60):
    """Get AI-powered performance predictions"""
    if not monitor.metric_history:
        raise HTTPException(status_code=404, detail="No metrics available for prediction")
        
    recent_metrics = monitor.metric_history[-500:]  # Last 500 points
    predictions = monitor.ai_analyzer.predict_future_performance(recent_metrics, horizon_minutes)
    
    return {
        "timestamp": time.time(),
        "predictions": predictions,
        "horizon_minutes": horizon_minutes
    }

@app.get("/api/dashboard/charts")
async def get_dashboard_charts():
    """Generate chart data for dashboard"""
    if not monitor.metric_history:
        return {"charts": []}
        
    current_time = time.time()
    last_hour_metrics = [
        metric for metric in monitor.metric_history
        if current_time - metric.timestamp <= 3600
    ]
    
    # Group metrics by component and type
    chart_data = {}
    
    for metric in last_hour_metrics:
        key = f"{metric.component}_{metric.metric_name}"
        if key not in chart_data:
            chart_data[key] = {
                "timestamps": [],
                "values": [],
                "component": metric.component,
                "metric_name": metric.metric_name,
                "unit": metric.unit
            }
            
        chart_data[key]["timestamps"].append(
            datetime.fromtimestamp(metric.timestamp).strftime("%H:%M:%S")
        )
        chart_data[key]["values"].append(metric.value)
        
    return {
        "timestamp": current_time,
        "charts": list(chart_data.values())
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the dashboard HTML page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI Performance Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #333; }
        .metric-label { color: #666; margin-bottom: 10px; }
        .status-normal { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alerts-panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .alert-item { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .alert-critical { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .charts-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .connection-status { position: fixed; top: 10px; right: 10px; padding: 5px 10px; border-radius: 5px; color: white; }
        .connected { background-color: #28a745; }
        .disconnected { background-color: #dc3545; }
        .insights-panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }
    </style>
</head>
<body>
    <div class="connection-status disconnected" id="connectionStatus">Disconnected</div>
    
    <div class="header">
        <h1>🚀 SutazAI Performance Monitoring Dashboard</h1>
        <p>Real-time system performance monitoring with AI-powered insights</p>
    </div>
    
    <div class="metrics-grid" id="metricsGrid">
        <!-- Metrics will be populated here -->
    </div>
    
    <div class="alerts-panel">
        <h3>🚨 Active Alerts</h3>
        <div id="alertsList">No active alerts</div>
    </div>
    
    <div class="insights-panel">
        <h3>🤖 AI Performance Insights</h3>
        <div id="aiInsights">Loading insights...</div>
    </div>
    
    <div class="charts-container">
        <div class="chart-container">
            <h3>System CPU Usage</h3>
            <div id="cpuChart"></div>
        </div>
        <div class="chart-container">
            <h3>Memory Usage</h3>
            <div id="memoryChart"></div>
        </div>
        <div class="chart-container">
            <h3>API Response Times</h3>
            <div id="responseTimeChart"></div>
        </div>
        <div class="chart-container">
            <h3>System Load Trends</h3>
            <div id="loadChart"></div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        let ws = null;
        let reconnectInterval = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                document.getElementById('connectionStatus').textContent = 'Connected';
                document.getElementById('connectionStatus').className = 'connection-status connected';
                
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'metrics_update') {
                    updateMetricsDisplay(data.metrics);
                } else if (data.type === 'ai_insights') {
                    updateAIInsights(data.insights);
                } else if (data.type === 'initial_metrics') {
                    updateMetricsDisplay(data.metrics);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                
                // Attempt to reconnect
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(() => {
                        console.log('Attempting to reconnect...');
                        connectWebSocket();
                    }, 5000);
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateMetricsDisplay(metrics) {
            const metricsGrid = document.getElementById('metricsGrid');
            const metricsByComponent = {};
            
            // Group metrics by component
            metrics.forEach(metric => {
                if (!metricsByComponent[metric.component]) {
                    metricsByComponent[metric.component] = {};
                }
                metricsByComponent[metric.component][metric.metric_name] = metric;
            });
            
            // Generate metric cards
            let html = '';
            Object.keys(metricsByComponent).forEach(component => {
                Object.keys(metricsByComponent[component]).forEach(metricName => {
                    const metric = metricsByComponent[component][metricName];
                    const statusClass = `status-${metric.status}`;
                    
                    html += `
                        <div class="metric-card">
                            <div class="metric-label">${component} - ${metricName}</div>
                            <div class="metric-value ${statusClass}">
                                ${metric.value.toFixed(1)} ${metric.unit}
                            </div>
                            <div style="font-size: 0.8em; color: #999;">
                                ${new Date(metric.timestamp * 1000).toLocaleTimeString()}
                            </div>
                        </div>
                    `;
                });
            });
            
            metricsGrid.innerHTML = html;
            
            // Update charts
            updateCharts(metrics);
        }
        
        function updateCharts(metrics) {
            // Filter and prepare data for different charts
            const cpuMetrics = metrics.filter(m => m.metric_name === 'cpu_usage' && m.component === 'system');
            const memoryMetrics = metrics.filter(m => m.metric_name === 'memory_usage' && m.component === 'system');
            const responseTimeMetrics = metrics.filter(m => m.metric_name.includes('response_time'));
            
            // CPU Chart
            if (cpuMetrics.length > 0) {
                const trace = {
                    x: cpuMetrics.map(m => new Date(m.timestamp * 1000)),
                    y: cpuMetrics.map(m => m.value),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'CPU Usage %',
                    line: { color: '#667eea' }
                };
                
                Plotly.newPlot('cpuChart', [trace], {
                    margin: { t: 0, r: 0, b: 30, l: 40 },
                    height: 200,
                    xaxis: { type: 'date' },
                    yaxis: { title: 'CPU %', range: [0, 100] }
                });
            }
            
            // Memory Chart
            if (memoryMetrics.length > 0) {
                const trace = {
                    x: memoryMetrics.map(m => new Date(m.timestamp * 1000)),
                    y: memoryMetrics.map(m => m.value),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Memory Usage %',
                    line: { color: '#764ba2' }
                };
                
                Plotly.newPlot('memoryChart', [trace], {
                    margin: { t: 0, r: 0, b: 30, l: 40 },
                    height: 200,
                    xaxis: { type: 'date' },
                    yaxis: { title: 'Memory %', range: [0, 100] }
                });
            }
            
            // Response Time Chart
            if (responseTimeMetrics.length > 0) {
                const traces = {};
                responseTimeMetrics.forEach(metric => {
                    const endpoint = metric.metadata?.endpoint || 'unknown';
                    if (!traces[endpoint]) {
                        traces[endpoint] = {
                            x: [],
                            y: [],
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: endpoint
                        };
                    }
                    traces[endpoint].x.push(new Date(metric.timestamp * 1000));
                    traces[endpoint].y.push(metric.value);
                });
                
                Plotly.newPlot('responseTimeChart', Object.values(traces), {
                    margin: { t: 0, r: 0, b: 30, l: 40 },
                    height: 200,
                    xaxis: { type: 'date' },
                    yaxis: { title: 'Response Time (ms)' }
                });
            }
        }
        
        function updateAIInsights(insights) {
            const insightsDiv = document.getElementById('aiInsights');
            
            let html = `<p><strong>Analysis Status:</strong> ${insights.status}</p>`;
            
            if (insights.insights && insights.insights.length > 0) {
                html += '<h4>🔍 Key Insights:</h4><ul>';
                insights.insights.forEach(insight => {
                    html += `<li>${insight}</li>`;
                });
                html += '</ul>';
            }
            
            if (insights.anomalies && insights.anomalies.length > 0) {
                html += '<h4>⚠️ Anomalies Detected:</h4><ul>';
                insights.anomalies.forEach(anomaly => {
                    html += `<li>${anomaly.metric}: ${anomaly.anomaly_count} anomalies (${anomaly.anomaly_percentage.toFixed(1)}%)</li>`;
                });
                html += '</ul>';
            }
            
            insightsDiv.innerHTML = html;
        }
        
        function loadActiveAlerts() {
            fetch('/api/alerts/active')
                .then(response => response.json())
                .then(data => {
                    const alertsList = document.getElementById('alertsList');
                    
                    if (data.alerts.length === 0) {
                        alertsList.innerHTML = 'No active alerts';
                        return;
                    }
                    
                    let html = '';
                    data.alerts.forEach(alert => {
                        html += `
                            <div class="alert-item alert-${alert.severity}">
                                <strong>${alert.severity.toUpperCase()}:</strong> ${alert.message}
                                <br><small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                            </div>
                        `;
                    });
                    
                    alertsList.innerHTML = html;
                })
                .catch(error => {
                    console.error('Error loading alerts:', error);
                });
        }
        
        function loadAIInsights() {
            fetch('/api/analysis/ai-insights')
                .then(response => response.json())
                .then(data => {
                    updateAIInsights(data.analysis);
                })
                .catch(error => {
                    console.error('Error loading AI insights:', error);
                    document.getElementById('aiInsights').innerHTML = 'Error loading insights';
                });
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            loadActiveAlerts();
            loadAIInsights();
            
            // Refresh alerts and insights periodically
            setInterval(loadActiveAlerts, 30000); // Every 30 seconds
            setInterval(loadAIInsights, 300000);  // Every 5 minutes
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "performance_monitoring_dashboard:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )