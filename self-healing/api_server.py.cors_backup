#!/usr/bin/env python3
"""
Self-Healing Orchestrator API Server

Provides REST API endpoints for monitoring and controlling the hygiene orchestrator.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from hygiene_orchestrator import SelfHealingOrchestrator, HygieneDatabase, ViolationPattern

# Pydantic models for API
class ViolationResponse(BaseModel):
    pattern_type: str
    severity: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    auto_fixable: bool
    risk_level: str
    detected_at: datetime

class FixActionResponse(BaseModel):
    action_id: str
    action_type: str
    source_path: str
    target_path: Optional[str] = None
    executed_at: Optional[datetime] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None

class SystemHealthResponse(BaseModel):
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    violation_count: int
    fix_success_rate: float
    git_status: str
    ci_status: str

class ScanRequest(BaseModel):
    force: bool = False
    dry_run: bool = False
    max_fixes: Optional[int] = None

class ConfigUpdateRequest(BaseModel):
    auto_fix_enabled: Optional[bool] = None
    dry_run: Optional[bool] = None
    scan_interval: Optional[int] = None
    risk_threshold: Optional[str] = None
    max_fixes_per_scan: Optional[int] = None

# Global orchestrator instance
orchestrator: Optional[SelfHealingOrchestrator] = None

app = FastAPI(
    title="Self-Healing Orchestrator API",
    description="API for monitoring and controlling codebase hygiene",
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

def get_orchestrator() -> SelfHealingOrchestrator:
    """Dependency to get orchestrator instance"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "Self-Healing Orchestrator API",
        "version": "1.0.0",
        "status": "running" if orchestrator and orchestrator.running else "stopped",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "violations": "/violations",
            "scan": "/scan",
            "config": "/config",
            "dashboard": "/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = orchestrator.get_status()
    
    return {
        "status": "healthy" if status["running"] else "unhealthy",
        "last_scan": status["last_scan_time"],
        "uptime": status["running"],
        "checks": {
            "orchestrator_running": status["running"],
        }
    }

@app.get("/status", response_model=Dict[str, Any])
async def get_status(orch: SelfHealingOrchestrator = Depends(get_orchestrator)):
    """Get current orchestrator status"""
    return orch.get_status()

@app.get("/violations", response_model=List[ViolationResponse])
async def get_violations(
    hours: int = Query(24, description="Hours to look back for violations"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Get recent violations"""
    violations = orch.database.get_recent_violations(hours)
    
    # Apply filters
    if severity:
        violations = [v for v in violations if v['severity'] == severity]
    if pattern_type:
        violations = [v for v in violations if v['pattern_type'] == pattern_type]
    
    return violations

@app.get("/violations/summary")
async def get_violations_summary(
    hours: int = Query(24, description="Hours to look back"),
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Get violations summary statistics"""
    violations = orch.database.get_recent_violations(hours)
    
    # Group by severity
    by_severity = {}
    by_type = {}
    
    for violation in violations:
        severity = violation['severity']
        pattern_type = violation['pattern_type']
        
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_type[pattern_type] = by_type.get(pattern_type, 0) + 1
    
    return {
        "total_violations": len(violations),
        "by_severity": by_severity,
        "by_type": by_type,
        "period_hours": hours,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/scan")
async def trigger_scan(
    request: ScanRequest = ScanRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Trigger a manual scan"""
    
    # For immediate response, run quick scan
    if not request.force:
        result = orch.manual_scan()
        return {
            "scan_triggered": True,
            "immediate_result": result,
            "message": "Quick scan completed"
        }
    
    # For full scan, run in background
    async def run_full_scan():
        try:
            # Temporarily override settings if requested
            original_dry_run = orch.config.get('dry_run', False)
            original_max_fixes = orch.config.get('max_fixes_per_scan', 50)
            
            if request.dry_run is not None:
                orch.config['dry_run'] = request.dry_run
                orch.fix_engine.set_dry_run(request.dry_run)
            
            if request.max_fixes is not None:
                orch.config['max_fixes_per_scan'] = request.max_fixes
            
            # Run scan
            await orch._scan_and_heal()
            
            # Restore original settings
            orch.config['dry_run'] = original_dry_run
            orch.config['max_fixes_per_scan'] = original_max_fixes
            orch.fix_engine.set_dry_run(original_dry_run)
            
        except Exception as e:
            logging.error(f"Background scan failed: {e}")
    
    background_tasks.add_task(run_full_scan)
    
    return {
        "scan_triggered": True,
        "message": "Full scan started in background",
        "settings": {
            "dry_run": request.dry_run,
            "max_fixes": request.max_fixes
        }
    }

@app.get("/config")
async def get_config(orch: SelfHealingOrchestrator = Depends(get_orchestrator)):
    """Get current configuration"""
    # Return only safe configuration values (no secrets)
    safe_config = {
        "auto_fix_enabled": orch.config.get("auto_fix_enabled"),
        "dry_run": orch.config.get("dry_run"),
        "scan_interval": orch.config.get("scan_interval"),
        "risk_threshold": orch.config.get("risk_threshold"),
        "max_fixes_per_scan": orch.config.get("max_fixes_per_scan"),
        "notification_enabled": orch.config.get("notification_enabled"),
        "project_root": orch.config.get("project_root"),
        "backup_dir": orch.config.get("backup_dir")
    }
    return safe_config

@app.put("/config")
async def update_config(
    request: ConfigUpdateRequest,
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Update configuration"""
    updated_fields = {}
    
    if request.auto_fix_enabled is not None:
        orch.config['auto_fix_enabled'] = request.auto_fix_enabled
        updated_fields['auto_fix_enabled'] = request.auto_fix_enabled
    
    if request.dry_run is not None:
        orch.config['dry_run'] = request.dry_run
        orch.fix_engine.set_dry_run(request.dry_run)
        updated_fields['dry_run'] = request.dry_run
    
    if request.scan_interval is not None:
        if request.scan_interval < 60:
            raise HTTPException(status_code=400, detail="Scan interval must be at least 60 seconds")
        orch.config['scan_interval'] = request.scan_interval
        orch.scan_interval = request.scan_interval
        updated_fields['scan_interval'] = request.scan_interval
    
    if request.risk_threshold is not None:
        if request.risk_threshold not in ['low', 'medium', 'high']:
            raise HTTPException(status_code=400, detail="Risk threshold must be 'low', 'medium', or 'high'")
        orch.config['risk_threshold'] = request.risk_threshold
        updated_fields['risk_threshold'] = request.risk_threshold
    
    if request.max_fixes_per_scan is not None:
        if request.max_fixes_per_scan < 1 or request.max_fixes_per_scan > 1000:
            raise HTTPException(status_code=400, detail="Max fixes per scan must be between 1 and 1000")
        orch.config['max_fixes_per_scan'] = request.max_fixes_per_scan
        updated_fields['max_fixes_per_scan'] = request.max_fixes_per_scan
    
    return {
        "message": "Configuration updated successfully",
        "updated_fields": updated_fields
    }

@app.get("/metrics")
async def get_metrics(
    hours: int = Query(24, description="Hours to look back"),
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Get system metrics for monitoring"""
    # Get health metrics from database
    import sqlite3
    with sqlite3.connect(orch.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM system_health 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours))
        
        health_metrics = [dict(row) for row in cursor.fetchall()]
    
    # Calculate aggregated metrics
    if health_metrics:
        latest = health_metrics[0]
        avg_cpu = sum(m['cpu_usage'] for m in health_metrics) / len(health_metrics)
        avg_memory = sum(m['memory_usage'] for m in health_metrics) / len(health_metrics)
        avg_disk = sum(m['disk_usage'] for m in health_metrics) / len(health_metrics)
    else:
        latest = {}
        avg_cpu = avg_memory = avg_disk = 0
    
    return {
        "current": latest,
        "averages": {
            "cpu_usage": avg_cpu,
            "memory_usage": avg_memory,
            "disk_usage": avg_disk
        },
        "history": health_metrics[:100],  # Limit to last 100 points
        "period_hours": hours
    }

@app.get("/fixes")
async def get_fix_history(
    hours: int = Query(24, description="Hours to look back"),
    success_only: bool = Query(False, description="Show only successful fixes"),
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Get fix action history"""
    # Get fix actions from database
    import sqlite3
    with sqlite3.connect(orch.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        where_clause = "WHERE executed_at > datetime('now', '-{} hours')".format(hours)
        if success_only:
            where_clause += " AND success = 1"
        
        cursor.execute(f"""
            SELECT fa.*, v.description as violation_description, v.severity
            FROM fix_actions fa
            LEFT JOIN violations v ON fa.violation_id = v.id
            {where_clause}
            ORDER BY executed_at DESC
        """)
        
        fixes = [dict(row) for row in cursor.fetchall()]
    
    return {
        "fixes": fixes,
        "total_count": len(fixes),
        "success_count": sum(1 for f in fixes if f['success']),
        "period_hours": hours
    }

@app.post("/fixes/{action_id}/rollback")
async def rollback_fix(
    action_id: str,
    orch: SelfHealingOrchestrator = Depends(get_orchestrator)
):
    """Rollback a specific fix action"""
    success = orch.fix_engine.rollback_fix(action_id)
    
    if success:
        return {"message": f"Fix {action_id} rolled back successfully"}
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"Fix {action_id} not found or cannot be rolled back"
        )

@app.get("/predictions")
async def get_predictions(orch: SelfHealingOrchestrator = Depends(get_orchestrator)):
    """Get current system predictions"""
    # Get current health
    current_health = orch._collect_health_metrics()
    
    # Get predictions
    predictions = orch.health_monitor.predict_issues(current_health)
    
    return {
        "predictions": predictions,
        "current_health": {
            "cpu_usage": current_health.cpu_usage,
            "memory_usage": current_health.memory_usage,
            "disk_usage": current_health.disk_usage,
            "violation_count": current_health.violation_count,
            "fix_success_rate": current_health.fix_success_rate
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve a simple monitoring dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Self-Healing Orchestrator Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-card { 
                display: inline-block; 
                margin: 10px; 
                padding: 20px; 
                border: 1px solid #ddd; 
                border-radius: 5px;
                min-width: 200px;
            }
            .metric-value { font-size: 24px; font-weight: bold; }
            .metric-label { color: #666; }
            .status-running { color: green; }
            .status-stopped { color: red; }
            .severity-critical { color: red; }
            .severity-high { color: orange; }
            .severity-medium { color: yellow; }
            .severity-low { color: green; }
        </style>
    </head>
    <body>
        <h1>🔧 Self-Healing Orchestrator Dashboard</h1>
        
        <div id="status-cards">
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value" id="status">Loading...</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Violations (24h)</div>
                <div class="metric-value" id="violations">Loading...</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fix Success Rate</div>
                <div class="metric-value" id="success-rate">Loading...</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last Scan</div>
                <div class="metric-value" id="last-scan">Loading...</div>
            </div>
        </div>
        
        <h2>📊 Violations by Severity</h2>
        <canvas id="violations-chart" width="400" height="200"></canvas>
        
        <h2>📈 System Metrics</h2>
        <canvas id="system-chart" width="400" height="200"></canvas>
        
        <h2>🔧 Recent Actions</h2>
        <button onclick="triggerScan()">Trigger Manual Scan</button>
        <button onclick="toggleDryRun()">Toggle Dry Run</button>
        <div id="recent-actions"></div>
        
        <script>
            let dryRun = false;
            
            async function loadStatus() {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    
                    document.getElementById('status').textContent = status.running ? 'Running' : 'Stopped';
                    document.getElementById('status').className = status.running ? 'status-running' : 'status-stopped';
                    document.getElementById('violations').textContent = status.recent_violations;
                    document.getElementById('success-rate').textContent = (status.fix_success_rate * 100).toFixed(1) + '%';
                    document.getElementById('last-scan').textContent = status.last_scan_time ? 
                        new Date(status.last_scan_time).toLocaleString() : 'Never';
                } catch (error) {
                    console.error('Failed to load status:', error);
                }
            }
            
            async function loadViolationsChart() {
                try {
                    const response = await fetch('/violations/summary');
                    const data = await response.json();
                    
                    const ctx = document.getElementById('violations-chart').getContext('2d');
                    new Chart(ctx, {
                        type: 'pie',
                        data: {
                            labels: Object.keys(data.by_severity),
                            datasets: [{
                                data: Object.values(data.by_severity),
                                backgroundColor: ['#ff6384', '#ff9f40', '#ffcd56', '#4bc0c0']
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Violations by Severity (Last 24h)'
                                }
                            }
                        }
                    });
                } catch (error) {
                    console.error('Failed to load violations chart:', error);
                }
            }
            
            async function loadSystemChart() {
                try {
                    const response = await fetch('/metrics?hours=6');
                    const data = await response.json();
                    
                    const ctx = document.getElementById('system-chart').getContext('2d');
                    const labels = data.history.map(h => new Date(h.timestamp).toLocaleTimeString()).reverse();
                    
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: 'CPU %',
                                    data: data.history.map(h => h.cpu_usage).reverse(),
                                    borderColor: '#ff6384',
                                    fill: false
                                },
                                {
                                    label: 'Memory %',
                                    data: data.history.map(h => h.memory_usage).reverse(),
                                    borderColor: '#4bc0c0',
                                    fill: false
                                },
                                {
                                    label: 'Disk %',
                                    data: data.history.map(h => h.disk_usage).reverse(),
                                    borderColor: '#ffcd56',
                                    fill: false
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'System Resources (Last 6h)'
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100
                                }
                            }
                        }
                    });
                } catch (error) {
                    console.error('Failed to load system chart:', error);
                }
            }
            
            async function triggerScan() {
                try {
                    const response = await fetch('/scan', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            force: true,
                            dry_run: dryRun
                        })
                    });
                    const result = await response.json();
                    alert('Scan triggered: ' + result.message);
                } catch (error) {
                    alert('Failed to trigger scan: ' + error);
                }
            }
            
            async function toggleDryRun() {
                dryRun = !dryRun;
                try {
                    const response = await fetch('/config', {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            dry_run: dryRun
                        })
                    });
                    alert('Dry run ' + (dryRun ? 'enabled' : 'disabled'));
                } catch (error) {
                    alert('Failed to toggle dry run: ' + error);
                }
            }
            
            // Load data on page load
            loadStatus();
            loadViolationsChart();
            loadSystemChart();
            
            // Refresh data every 30 seconds
            setInterval(loadStatus, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    try:
        orchestrator = SelfHealingOrchestrator()
        logging.info("Orchestrator initialized successfully")
        
        # Start orchestrator in background
        asyncio.create_task(orchestrator.start())
        
    except Exception as e:
        logging.error(f"Failed to initialize orchestrator: {e}")
        raise

async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator
    if orchestrator:
        orchestrator.stop()
        logging.info("Orchestrator stopped")

# Register startup and shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

def run_server(host: str = "0.0.0.0", port: int = 8010, debug: bool = False):
    """Run the API server"""
    log_level = "debug" if debug else "info"
    uvicorn.run(
        "api_server:app", 
        host=host, 
        port=port, 
        log_level=log_level,
        reload=debug
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Healing Orchestrator API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.debug)