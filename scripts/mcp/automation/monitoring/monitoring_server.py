#!/usr/bin/env python3
"""
MCP Automation Monitoring HTTP Server
Unified monitoring service exposing metrics, health, and dashboards
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST
import yaml

# Import monitoring components
from metrics_collector import MCPMetricsCollector
from health_monitor import MCPHealthMonitor, HealthStatus
from alert_manager import AlertManager, AlertSeverity
from dashboard_config import DashboardManager
from log_aggregator import LogAggregator, LogLevel, LogSource
from sla_monitor import SLAMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MCP Automation Monitoring",
    description="Comprehensive monitoring system for MCP automation infrastructure",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global monitoring components
metrics_collector: Optional[MCPMetricsCollector] = None
health_monitor: Optional[MCPHealthMonitor] = None
alert_manager: Optional[AlertManager] = None
dashboard_manager: Optional[DashboardManager] = None
log_aggregator: Optional[LogAggregator] = None
sla_monitor: Optional[SLAMonitor] = None

# WebSocket connections for real-time updates
websocket_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring components on startup"""
    global metrics_collector, health_monitor, alert_manager
    global dashboard_manager, log_aggregator, sla_monitor
    
    logger.info("Starting MCP Automation Monitoring Server...")
    
    # Initialize components
    metrics_collector = MCPMetricsCollector(
        push_gateway=os.getenv("PROMETHEUS_PUSHGATEWAY", "http://localhost:10200/metrics")
    )
    
    health_monitor = MCPHealthMonitor(
        check_interval=30
    )
    
    alert_manager = AlertManager(
        correlation_window_minutes=5
    )
    
    dashboard_manager = DashboardManager(
        grafana_url=os.getenv("GRAFANA_URL", "http://localhost:10201")
    )
    
    log_aggregator = LogAggregator(
        loki_url=os.getenv("LOKI_URL", "http://localhost:10202")
    )
    
    sla_monitor = SLAMonitor()
    
    # Start background tasks
    asyncio.create_task(metrics_collector.start_collection_loop())
    asyncio.create_task(health_monitor.start_monitoring_loop())
    asyncio.create_task(log_aggregator.start_aggregation_loop())
    asyncio.create_task(alert_processing_loop())
    asyncio.create_task(sla_monitoring_loop())
    asyncio.create_task(websocket_broadcast_loop())
    
    # Deploy dashboards to Grafana
    asyncio.create_task(dashboard_manager.deploy_all_dashboards())
    
    logger.info("Monitoring components initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down monitoring server...")
    
    if health_monitor:
        await health_monitor.cleanup()
    if alert_manager:
        await alert_manager.cleanup()
    if dashboard_manager:
        await dashboard_manager.cleanup()
    if log_aggregator:
        await log_aggregator.cleanup()
        
    # Close WebSocket connections
    for ws in websocket_connections:
        await ws.close()


# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/health/detailed")
async def detailed_health():
    """Detailed health status of all components"""
    if not health_monitor:
        raise HTTPException(status_code=503, detail="Health monitor not initialized")
        
    health_status = await health_monitor.perform_health_checks()
    return health_monitor.get_health_summary()


# Metrics endpoints
@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
        
    metrics = metrics_collector.get_metrics()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary")
async def metrics_summary():
    """Human-readable metrics summary"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
        
    return metrics_collector.get_metrics_summary()


# Alert endpoints
@app.get("/alerts")
async def get_alerts():
    """Get active alerts"""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
        
    alerts = alert_manager.get_active_alerts()
    return {
        "alerts": [
            {
                "id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "state": alert.state.value,
                "component": alert.component,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in alerts
        ],
        "statistics": alert_manager.get_alert_statistics()
    }


@app.post("/alerts")
async def create_alert(alert_data: Dict[str, Any]):
    """Create a new alert"""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
        
    try:
        alert = await alert_manager.create_alert(
            name=alert_data.get("name", "manual_alert"),
            severity=AlertSeverity(alert_data.get("severity", "warning")),
            component=alert_data.get("component", "unknown"),
            message=alert_data.get("message", "Manual alert"),
            details=alert_data.get("details", {})
        )
        
        return {
            "alert_id": alert.id,
            "status": "created",
            "state": alert.state.value
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: str = "admin"):
    """Acknowledge an alert"""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
        
    await alert_manager.acknowledge_alert(alert_id, user)
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not initialized")
        
    await alert_manager.resolve_alert(alert_id)
    return {"status": "resolved", "alert_id": alert_id}


# SLA endpoints
@app.get("/sla/status")
async def sla_status():
    """Get current SLA status"""
    if not sla_monitor:
        raise HTTPException(status_code=503, detail="SLA monitor not initialized")
        
    return sla_monitor.get_current_status()


@app.get("/sla/report")
async def sla_report(
    start_time: Optional[str] = Query(None, description="Start time in ISO format"),
    end_time: Optional[str] = Query(None, description="End time in ISO format")
):
    """Generate SLA compliance report"""
    if not sla_monitor:
        raise HTTPException(status_code=503, detail="SLA monitor not initialized")
        
    start = datetime.fromisoformat(start_time) if start_time else None
    end = datetime.fromisoformat(end_time) if end_time else None
    
    report = sla_monitor.generate_sla_report(start, end)
    
    return {
        "period_start": report.period_start.isoformat(),
        "period_end": report.period_end.isoformat(),
        "overall_compliance": report.overall_compliance,
        "violations": report.violations,
        "recommendations": report.recommendations,
        "generated_at": report.generated_at.isoformat()
    }


@app.post("/sla/measurement")
async def record_sla_measurement(measurement_data: Dict[str, Any]):
    """Record an SLA measurement"""
    if not sla_monitor:
        raise HTTPException(status_code=503, detail="SLA monitor not initialized")
        
    try:
        measurement = sla_monitor.record_measurement(
            slo_name=measurement_data["slo_name"],
            value=float(measurement_data["value"]),
            timestamp=datetime.fromisoformat(measurement_data["timestamp"]) 
                if "timestamp" in measurement_data else None
        )
        
        return {
            "status": "recorded",
            "compliance_status": measurement.compliance_status.value,
            "error_budget_consumed": measurement.error_budget_consumed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Log endpoints
@app.post("/logs")
async def ingest_log(log_data: Dict[str, Any]):
    """Ingest a log entry"""
    if not log_aggregator:
        raise HTTPException(status_code=503, detail="Log aggregator not initialized")
        
    try:
        from log_aggregator import LogEntry
        
        log_entry = LogEntry(
            timestamp=datetime.fromisoformat(log_data.get("timestamp", datetime.now().isoformat())),
            level=LogLevel(log_data.get("level", "info")),
            source=LogSource(log_data.get("source", "system")),
            component=log_data.get("component", "unknown"),
            message=log_data.get("message", ""),
            structured_data=log_data.get("data", {}),
            trace_id=log_data.get("trace_id"),
            span_id=log_data.get("span_id"),
            user=log_data.get("user"),
            tags=log_data.get("tags", [])
        )
        
        await log_aggregator.ingest_log(log_entry)
        
        return {"status": "ingested"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/logs/search")
async def search_logs(
    query: str,
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 100
):
    """Search logs"""
    if not log_aggregator:
        raise HTTPException(status_code=503, detail="Log aggregator not initialized")
        
    log_level = LogLevel(level) if level else None
    
    results = log_aggregator.search_logs(
        query=query,
        level=log_level,
        component=component,
        limit=limit
    )
    
    return {
        "results": [
            {
                "timestamp": log.timestamp.isoformat(),
                "level": log.level.value,
                "component": log.component,
                "message": log.message,
                "tags": log.tags
            }
            for log in results
        ],
        "count": len(results)
    }


@app.get("/logs/analysis")
async def log_analysis():
    """Get log analysis and error patterns"""
    if not log_aggregator:
        raise HTTPException(status_code=503, detail="Log aggregator not initialized")
        
    return {
        "summary": log_aggregator.get_aggregation_summary(),
        "error_analysis": log_aggregator.get_error_analysis()
    }


# Dashboard endpoints
@app.post("/dashboards/deploy/{dashboard_key}")
async def deploy_dashboard(dashboard_key: str):
    """Deploy a dashboard to Grafana"""
    if not dashboard_manager:
        raise HTTPException(status_code=503, detail="Dashboard manager not initialized")
        
    success = await dashboard_manager.deploy_dashboard(dashboard_key)
    
    if success:
        return {"status": "deployed", "dashboard": dashboard_key}
    else:
        raise HTTPException(status_code=500, detail="Failed to deploy dashboard")


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


# Background tasks
async def alert_processing_loop():
    """Process alert notifications"""
    while True:
        try:
            if alert_manager:
                await alert_manager.process_notifications()
        except Exception as e:
            logger.error(f"Error in alert processing: {e}")
            
        await asyncio.sleep(5)


async def sla_monitoring_loop():
    """Monitor SLA compliance"""
    while True:
        try:
            if sla_monitor and metrics_collector:
                # Get metrics summary
                metrics_summary = metrics_collector.get_metrics_summary()
                
                # Record SLA measurements based on metrics
                if 'mcp_servers' in metrics_summary:
                    availability = metrics_summary['mcp_servers'].get('health_percentage', 0) / 100
                    sla_monitor.record_measurement('mcp_availability', availability)
                    
                if 'system' in metrics_summary:
                    cpu = metrics_summary['system'].get('cpu_percent', 0)
                    sla_monitor.record_measurement('resource_utilization', cpu)
                    
        except Exception as e:
            logger.error(f"Error in SLA monitoring: {e}")
            
        await asyncio.sleep(60)


async def websocket_broadcast_loop():
    """Broadcast updates to WebSocket clients"""
    while True:
        try:
            if websocket_connections:
                # Prepare update data
                update = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "monitoring_update"
                }
                
                if health_monitor:
                    update["health"] = health_monitor.get_health_summary()
                    
                if alert_manager:
                    update["alerts"] = alert_manager.get_alert_statistics()
                    
                if sla_monitor:
                    update["sla"] = sla_monitor.get_current_status()
                    
                # Broadcast to all connected clients
                disconnected = []
                for ws in websocket_connections:
                    try:
                        await ws.send_json(update)
                    except:
                        disconnected.append(ws)
                        
                # Remove disconnected clients
                for ws in disconnected:
                    websocket_connections.remove(ws)
                    
        except Exception as e:
            logger.error(f"Error in WebSocket broadcast: {e}")
            
        await asyncio.sleep(10)


# HTML dashboard endpoint
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Automation Monitoring</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric { display: inline-block; margin: 10px 20px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .metric-label { color: #666; font-size: 14px; }
            .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
            .status.healthy { background: #d4edda; color: #155724; }
            .status.degraded { background: #fff3cd; color: #856404; }
            .status.unhealthy { background: #f8d7da; color: #721c24; }
            .links { margin: 20px 0; }
            .links a { margin: 0 10px; color: #007bff; text-decoration: none; }
            .links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 MCP Automation Monitoring Dashboard</h1>
            
            <div class="card">
                <h2>Quick Links</h2>
                <div class="links">
                    <a href="/docs">API Documentation</a>
                    <a href="/metrics">Prometheus Metrics</a>
                    <a href="/health/detailed">Health Status</a>
                    <a href="/alerts">Active Alerts</a>
                    <a href="/sla/status">SLA Status</a>
                    <a href="http://localhost:10201">Grafana Dashboards</a>
                </div>
            </div>
            
            <div class="card">
                <h2>System Status</h2>
                <div id="status">Loading...</div>
            </div>
            
            <div class="card">
                <h2>Real-time Metrics</h2>
                <div id="metrics">Connecting...</div>
            </div>
        </div>
        
        <script>
            // Fetch initial status
            fetch('/health/detailed')
                .then(res => res.json())
                .then(data => {
                    const statusClass = data.status === 'healthy' ? 'healthy' : 
                                       data.status === 'degraded' ? 'degraded' : 'unhealthy';
                    document.getElementById('status').innerHTML = `
                        <span class="status ${statusClass}">${data.status.toUpperCase()}</span>
                        <div class="metric">
                            <div class="metric-value">${data.healthy}</div>
                            <div class="metric-label">Healthy Components</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.degraded}</div>
                            <div class="metric-label">Degraded Components</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.unhealthy}</div>
                            <div class="metric-label">Unhealthy Components</div>
                        </div>
                    `;
                });
            
            // WebSocket for real-time updates
            const ws = new WebSocket('ws://localhost:10205/ws');
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.health) {
                    // Update metrics display
                    document.getElementById('metrics').innerHTML = `
                        <div class="metric">
                            <div class="metric-value">${data.alerts?.active_alerts || 0}</div>
                            <div class="metric-label">Active Alerts</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${(data.sla?.overall_health || 'unknown').toUpperCase()}</div>
                            <div class="metric-label">SLA Health</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${new Date(data.timestamp).toLocaleTimeString()}</div>
                            <div class="metric-label">Last Update</div>
                        </div>
                    `;
                }
            };
            
            ws.onerror = () => {
                document.getElementById('metrics').innerHTML = 'Connection error';
            };
        </script>
    </body>
    </html>
    """
    
    return html_content


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    sys.exit(0)


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get port from environment or default
    port = int(os.getenv("MONITORING_PORT", "10205"))
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()