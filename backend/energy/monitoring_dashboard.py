"""
Real-time Energy Monitoring Dashboard - Web-based energy consumption visualization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .energy_profiler import get_global_profiler, start_global_monitoring
from .power_optimizer import get_global_optimizer, OptimizationStrategy
from .agent_hibernation import get_hibernation_manager
from .workload_scheduler import get_global_scheduler, SchedulingPolicy
from .resource_allocator import get_global_allocator, AllocationStrategy

logger = logging.getLogger(__name__)

class EnergyMonitoringDashboard:
    """Real-time energy monitoring dashboard"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Initialize the energy monitoring dashboard
        
        Args:
            host: Host to bind the dashboard to
            port: Port to bind the dashboard to
        """
        self.host = host
        self.port = port
        self.app = FastAPI(title="SutazAI Energy Monitoring Dashboard")
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        
        # Initialize energy management components
        self.profiler = get_global_profiler()
        self.optimizer = get_global_optimizer()
        self.hibernation_manager = get_hibernation_manager()
        self.scheduler = get_global_scheduler()
        self.allocator = get_global_allocator()
        
        # Background task for broadcasting updates
        self._broadcasting = False
        self._broadcast_task: Optional[asyncio.Task] = None
        
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard HTML"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/energy/current")
        async def get_current_energy():
            """Get current energy metrics"""
            try:
                metrics = self.profiler.get_current_metrics()
                efficiency = self.profiler.get_efficiency_metrics()
                
                return JSONResponse({
                    "status": "success",
                    "data": {
                        "current_metrics": metrics,
                        "efficiency_metrics": efficiency,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.error(f"Error getting current energy metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/energy/history")
        async def get_energy_history(hours: float = 24.0):
            """Get energy consumption history"""
            try:
                energy_metrics = self.profiler.calculate_energy_metrics(hours)
                
                return JSONResponse({
                    "status": "success",
                    "data": {
                        "start_time": energy_metrics.start_time.isoformat(),
                        "end_time": energy_metrics.end_time.isoformat(),
                        "total_energy_wh": energy_metrics.total_energy_wh,
                        "avg_power_w": energy_metrics.avg_power_w,
                        "peak_power_w": energy_metrics.peak_power_w,
                        "cpu_energy_wh": energy_metrics.cpu_energy_wh,
                        "memory_energy_wh": energy_metrics.memory_energy_wh,
                        "co2_emission_g": energy_metrics.co2_emission_g,
                        "measurements": [
                            {
                                "timestamp": m.timestamp.isoformat(),
                                "total_power": m.total_power,
                                "cpu_power": m.cpu_power,
                                "memory_power": m.memory_power,
                                "cpu_utilization": m.cpu_utilization,
                                "memory_utilization": m.memory_utilization,
                                "active_agents": m.active_agents
                            }
                            for m in energy_metrics.measurements[-1000:]  # Last 1000 points
                        ]
                    }
                })
            except Exception as e:
                logger.error(f"Error getting energy history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/optimization/stats")
        async def get_optimization_stats():
            """Get power optimization statistics"""
            try:
                stats = self.optimizer.get_optimization_stats()
                return JSONResponse({
                    "status": "success",
                    "data": stats
                })
            except Exception as e:
                logger.error(f"Error getting optimization stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/optimization/strategy")
        async def set_optimization_strategy(strategy: str):
            """Set power optimization strategy"""
            try:
                if strategy not in [s.value for s in OptimizationStrategy]:
                    raise HTTPException(status_code=400, detail="Invalid strategy")
                
                # Stop current optimizer
                self.optimizer.stop_optimization()
                
                # Create new optimizer with new strategy
                self.optimizer = get_global_optimizer(OptimizationStrategy(strategy))
                self.optimizer.start_optimization()
                
                return JSONResponse({
                    "status": "success",
                    "message": f"Optimization strategy set to {strategy}"
                })
            except Exception as e:
                logger.error(f"Error setting optimization strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/hibernation/stats")
        async def get_hibernation_stats():
            """Get agent hibernation statistics"""
            try:
                stats = self.hibernation_manager.get_hibernation_stats()
                return JSONResponse({
                    "status": "success",
                    "data": stats
                })
            except Exception as e:
                logger.error(f"Error getting hibernation stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/hibernation/wake-all")
        async def wake_all_agents():
            """Wake all hibernated agents"""
            try:
                woken_count = self.hibernation_manager.force_wake_all()
                return JSONResponse({
                    "status": "success",
                    "message": f"Woke {woken_count} agents"
                })
            except Exception as e:
                logger.error(f"Error waking all agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/scheduling/stats")
        async def get_scheduling_stats():
            """Get workload scheduling statistics"""
            try:
                stats = self.scheduler.get_scheduling_stats()
                return JSONResponse({
                    "status": "success",
                    "data": stats
                })
            except Exception as e:
                logger.error(f"Error getting scheduling stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/allocation/stats")
        async def get_allocation_stats():
            """Get resource allocation statistics"""
            try:
                stats = self.allocator.get_allocation_stats()
                return JSONResponse({
                    "status": "success",
                    "data": stats
                })
            except Exception as e:
                logger.error(f"Error getting allocation stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/sustainability/metrics")
        async def get_sustainability_metrics():
            """Get comprehensive sustainability metrics"""
            try:
                # Get energy metrics
                energy_metrics = self.profiler.calculate_energy_metrics(24.0)
                
                # Get optimization stats
                opt_stats = self.optimizer.get_optimization_stats()
                
                # Get hibernation stats
                hibernation_stats = self.hibernation_manager.get_hibernation_stats()
                
                # Calculate sustainability scores
                sustainability_data = {
                    "energy_efficiency_score": self._calculate_energy_efficiency_score(energy_metrics),
                    "carbon_footprint": {
                        "daily_co2_g": energy_metrics.co2_emission_g,
                        "annual_co2_kg": energy_metrics.co2_emission_g * 365 / 1000,
                        "carbon_intensity_kg_kwh": 0.4  # Grid carbon intensity
                    },
                    "power_optimization": {
                        "total_power_saved_w": opt_stats.get("total_power_saved_w", 0.0),
                        "optimization_success_rate": opt_stats.get("success_rate", 0.0),
                        "active_optimizations": opt_stats.get("total_optimizations", 0)
                    },
                    "resource_efficiency": {
                        "hibernated_agents_ratio": hibernation_stats.get("hibernation_ratio", 0.0),
                        "hibernation_power_saved_w": hibernation_stats.get("total_power_saved_w", 0.0)
                    },
                    "sustainability_grade": self._calculate_sustainability_grade(
                        energy_metrics, opt_stats, hibernation_stats
                    )
                }
                
                return JSONResponse({
                    "status": "success",
                    "data": sustainability_data
                })
            except Exception as e:
                logger.error(f"Error getting sustainability metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket(websocket)
    
    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Keep connection alive and handle client messages
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
    
    async def _broadcast_updates(self) -> None:
        """Broadcast real-time updates to connected clients"""
        while self._broadcasting:
            try:
                if self.active_connections:
                    # Gather current data
                    update_data = {
                        "timestamp": datetime.now().isoformat(),
                        "energy": self.profiler.get_current_metrics(),
                        "efficiency": self.profiler.get_efficiency_metrics(),
                        "optimization": self.optimizer.get_optimization_stats(),
                        "hibernation": self.hibernation_manager.get_hibernation_stats(),
                        "scheduling": self.scheduler.get_scheduling_stats(),
                        "allocation": self.allocator.get_allocation_stats()
                    }
                    
                    # Send to all connected clients
                    disconnected = []
                    for connection in self.active_connections:
                        try:
                            await connection.send_text(json.dumps(update_data))
                        except:
                            disconnected.append(connection)
                    
                    # Remove disconnected clients
                    for connection in disconnected:
                        self.active_connections.remove(connection)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(5)
    
    def _calculate_energy_efficiency_score(self, energy_metrics) -> float:
        """Calculate energy efficiency score (0-100)"""
        # Base score on power consumption and utilization
        if energy_metrics.avg_power_w == 0:
            return 100.0
        
        # Ideal power consumption for the system (rough estimate)
        ideal_power_w = 20.0  # Minimum efficient power consumption
        
        # Score based on how close we are to ideal
        efficiency_ratio = ideal_power_w / max(energy_metrics.avg_power_w, ideal_power_w)
        score = min(100.0, efficiency_ratio * 100.0)
        
        return score
    
    def _calculate_sustainability_grade(self, energy_metrics, opt_stats, hibernation_stats) -> str:
        """Calculate overall sustainability grade (A-F)"""
        # Calculate composite score
        energy_score = self._calculate_energy_efficiency_score(energy_metrics)
        optimization_score = opt_stats.get("success_rate", 0.0) * 100
        hibernation_score = hibernation_stats.get("hibernation_ratio", 0.0) * 100
        
        # Weighted average
        composite_score = (
            energy_score * 0.4 +
            optimization_score * 0.3 +
            hibernation_score * 0.3
        )
        
        # Convert to letter grade
        if composite_score >= 90:
            return "A"
        elif composite_score >= 80:
            return "B"
        elif composite_score >= 70:
            return "C"
        elif composite_score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Energy Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 10px;
        }
        
        .metric-unit {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: #2ecc71; }
        .status-warning { background-color: #f39c12; }
        .status-offline { background-color: #e74c3c; }
        
        .controls {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
            transition: background 0.3s ease;
        }
        
        .btn:hover {
            background: #2980b9;
        }
        
        .btn-success { background: #2ecc71; }
        .btn-success:hover { background: #27ae60; }
        
        .btn-warning { background: #f39c12; }
        .btn-warning:hover { background: #e67e22; }
        
        .grade {
            display: inline-block;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            color: white;
            font-size: 1.5em;
            font-weight: bold;
            text-align: center;
            line-height: 60px;
            margin: 10px;
        }
        
        .grade-a { background: #2ecc71; }
        .grade-b { background: #3498db; }
        .grade-c { background: #f39c12; }
        .grade-d { background: #e67e22; }
        .grade-f { background: #e74c3c; }
        
        #connectionStatus {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="connectionStatus" class="status-offline">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>🌱 SutazAI Energy Monitoring Dashboard</h1>
            <p>Real-time energy consumption and sustainability metrics</p>
        </div>
        
        <div id="currentMetrics" class="metrics-grid">
            <!-- Metrics will be populated by JavaScript -->
        </div>
        
        <div class="controls">
            <h3>Controls</h3>
            <button class="btn btn-success" onclick="wakeAllAgents()">Wake All Agents</button>
            <button class="btn btn-warning" onclick="setOptimizationStrategy('aggressive')">Aggressive Power Saving</button>
            <button class="btn" onclick="setOptimizationStrategy('balanced')">Balanced Mode</button>
            <button class="btn" onclick="exportData()">Export Data</button>
        </div>
        
        <div class="chart-container">
            <h3>Power Consumption History</h3>
            <canvas id="powerChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Energy Efficiency Trend</h3>
            <canvas id="efficiencyChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Sustainability Overview</h3>
            <div id="sustainabilityMetrics">
                <!-- Sustainability metrics will be populated here -->
            </div>
        </div>
    </div>

    <script>
        let ws;
        let powerChart;
        let efficiencyChart;
        let powerData = [];
        let efficiencyData = [];
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                document.getElementById('connectionStatus').textContent = 'Connected';
                document.getElementById('connectionStatus').className = 'status-online';
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('connectionStatus').className = 'status-offline';
                console.log('WebSocket disconnected');
                
                // Attempt to reconnect after 5 seconds
                setTimeout(initWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                document.getElementById('connectionStatus').textContent = 'Connection Error';
                document.getElementById('connectionStatus').className = 'status-warning';
            };
        }
        
        // Update dashboard with new data
        function updateDashboard(data) {
            updateCurrentMetrics(data);
            updateCharts(data);
            updatePowerData(data.energy);
        }
        
        // Update current metrics display
        function updateCurrentMetrics(data) {
            const metrics = data.energy || {};
            const efficiency = data.efficiency || {};
            const optimization = data.optimization || {};
            const hibernation = data.hibernation || {};
            
            const metricsHtml = `
                <div class="metric-card">
                    <div class="metric-title">
                        <span class="status-indicator status-online"></span>
                        Current Power
                    </div>
                    <div class="metric-value">${(metrics.current_power_w || 0).toFixed(1)}</div>
                    <div class="metric-unit">Watts</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">CPU Utilization</div>
                    <div class="metric-value">${(metrics.cpu_utilization || 0).toFixed(1)}</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value">${(metrics.memory_utilization || 0).toFixed(1)}</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Active Agents</div>
                    <div class="metric-value">${metrics.active_agents || 0}</div>
                    <div class="metric-unit">agents</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Power Saved</div>
                    <div class="metric-value">${(optimization.total_power_saved_w || 0).toFixed(1)}</div>
                    <div class="metric-unit">Watts</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Hibernated Agents</div>
                    <div class="metric-value">${hibernation.currently_hibernated || 0}</div>
                    <div class="metric-unit">agents</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Efficiency Score</div>
                    <div class="metric-value">${(efficiency.utilization_efficiency_score || 0).toFixed(0)}</div>
                    <div class="metric-unit">/ 100</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">CPU Frequency</div>
                    <div class="metric-value">${(metrics.cpu_frequency_mhz || 0).toFixed(0)}</div>
                    <div class="metric-unit">MHz</div>
                </div>
            `;
            
            document.getElementById('currentMetrics').innerHTML = metricsHtml;
        }
        
        // Update power consumption data
        function updatePowerData(energyData) {
            if (!energyData) return;
            
            const now = new Date();
            const powerValue = energyData.current_power_w || 0;
            const efficiencyValue = energyData.utilization_efficiency_score || 0;
            
            powerData.push({
                x: now,
                y: powerValue
            });
            
            efficiencyData.push({
                x: now,
                y: efficiencyValue
            });
            
            // Keep only last 100 data points
            if (powerData.length > 100) {
                powerData.shift();
                efficiencyData.shift();
            }
            
            // Update charts
            if (powerChart) {
                powerChart.data.datasets[0].data = powerData;
                powerChart.update('none');
            }
            
            if (efficiencyChart) {
                efficiencyChart.data.datasets[0].data = efficiencyData;
                efficiencyChart.update('none');
            }
        }
        
        // Update charts
        function updateCharts(data) {
            // Charts are updated in updatePowerData function
        }
        
        // Initialize charts
        function initCharts() {
            const powerCtx = document.getElementById('powerChart').getContext('2d');
            powerChart = new Chart(powerCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Power Consumption (W)',
                        data: powerData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                displayFormats: {
                                    minute: 'HH:mm'
                                }
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Power (W)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
            
            const efficiencyCtx = document.getElementById('efficiencyChart').getContext('2d');
            efficiencyChart = new Chart(efficiencyCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Efficiency Score',
                        data: efficiencyData,
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                displayFormats: {
                                    minute: 'HH:mm'
                                }
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Efficiency Score'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
        }
        
        // Control functions
        async function wakeAllAgents() {
            try {
                const response = await fetch('/api/hibernation/wake-all', {
                    method: 'POST'
                });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error waking agents: ' + error.message);
            }
        }
        
        async function setOptimizationStrategy(strategy) {
            try {
                const response = await fetch('/api/optimization/strategy', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ strategy: strategy })
                });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error setting optimization strategy: ' + error.message);
            }
        }
        
        async function exportData() {
            try {
                const response = await fetch('/api/energy/history?hours=24');
                const data = await response.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {
                    type: 'application/json'
                });
                
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `energy-data-${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } catch (error) {
                alert('Error exporting data: ' + error.message);
            }
        }
        
        // Load sustainability metrics
        async function loadSustainabilityMetrics() {
            try {
                const response = await fetch('/api/sustainability/metrics');
                const result = await response.json();
                const data = result.data;
                
                const sustainabilityHtml = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                        <div style="text-align: center;">
                            <h4>Sustainability Grade</h4>
                            <div class="grade grade-${data.sustainability_grade.toLowerCase()}">${data.sustainability_grade}</div>
                        </div>
                        <div>
                            <h4>Carbon Footprint</h4>
                            <p><strong>Daily CO2:</strong> ${data.carbon_footprint.daily_co2_g.toFixed(1)} g</p>
                            <p><strong>Annual CO2:</strong> ${data.carbon_footprint.annual_co2_kg.toFixed(1)} kg</p>
                        </div>
                        <div>
                            <h4>Energy Efficiency</h4>
                            <p><strong>Score:</strong> ${data.energy_efficiency_score.toFixed(1)}/100</p>
                            <p><strong>Power Saved:</strong> ${data.power_optimization.total_power_saved_w.toFixed(1)} W</p>
                        </div>
                        <div>
                            <h4>Resource Efficiency</h4>
                            <p><strong>Hibernated:</strong> ${(data.resource_efficiency.hibernated_agents_ratio * 100).toFixed(1)}%</p>
                            <p><strong>Power Saved:</strong> ${data.resource_efficiency.hibernation_power_saved_w.toFixed(1)} W</p>
                        </div>
                    </div>
                `;
                
                document.getElementById('sustainabilityMetrics').innerHTML = sustainabilityHtml;
            } catch (error) {
                console.error('Error loading sustainability metrics:', error);
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            initWebSocket();
            loadSustainabilityMetrics();
            
            // Refresh sustainability metrics every 5 minutes
            setInterval(loadSustainabilityMetrics, 5 * 60 * 1000);
        });
    </script>
</body>
</html>
        """
    
    async def start_dashboard(self) -> None:
        """Start the energy monitoring dashboard"""
        # Start energy monitoring components
        start_global_monitoring()
        self.optimizer.start_optimization()
        self.hibernation_manager.start_monitoring()
        self.scheduler.start_scheduling()
        self.allocator.start_monitoring()
        
        # Start real-time broadcasting
        self._broadcasting = True
        self._broadcast_task = asyncio.create_task(self._broadcast_updates())
        
        logger.info(f"Starting energy monitoring dashboard on {self.host}:{self.port}")
        
        # Start the web server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop_dashboard(self) -> None:
        """Stop the energy monitoring dashboard"""
        # Stop broadcasting
        self._broadcasting = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
        
        # Stop energy monitoring components
        self.profiler.stop_monitoring()
        self.optimizer.stop_optimization()
        self.hibernation_manager.stop_monitoring()
        self.scheduler.stop_scheduling()
        self.allocator.stop_monitoring()
        
        logger.info("Energy monitoring dashboard stopped")

def create_dashboard(host: str = "0.0.0.0", port: int = 8080) -> EnergyMonitoringDashboard:
    """Create and return an energy monitoring dashboard instance"""
    return EnergyMonitoringDashboard(host, port)

async def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the energy monitoring dashboard"""
    dashboard = create_dashboard(host, port)
    await dashboard.start_dashboard()

if __name__ == "__main__":
    asyncio.run(run_dashboard())