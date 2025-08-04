#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for SutazAI Ollama Agents
Provides web-based real-time monitoring interface
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import sqlite3
import signal
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.ollama_agent_monitor import OllamaAgentMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RealtimeDashboard:
    """Real-time web dashboard for monitoring 131 Ollama agents"""
    
    def __init__(self, port: int = 8092):
        self.port = port
        self.app = web.Application()
        self.monitor = OllamaAgentMonitor()
        self.websocket_connections: set = set()
        self.shutdown_event = asyncio.Event()
        
        # Setup routes
        self._setup_routes()
        
        # CORS setup
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def _setup_routes(self):
        """Setup HTTP and WebSocket routes"""
        
        # Static file serving
        self.app.router.add_get('/', self.serve_dashboard)
        self.app.router.add_get('/health', self.health_check)
        
        # API endpoints
        self.app.router.add_get('/api/status', self.get_system_status)
        self.app.router.add_get('/api/agents', self.get_agents_status)
        self.app.router.add_get('/api/alerts', self.get_alerts)
        self.app.router.add_get('/api/metrics/history', self.get_metrics_history)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Static files
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if os.path.exists(static_dir):
            self.app.router.add_static('/', static_dir, name='static')
    
    async def serve_dashboard(self, request):
        """Serve the main dashboard HTML"""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML with embedded JavaScript"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI - Ollama Agent Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 1rem;
            text-align: center;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        
        .header .subtitle {
            font-size: 1.2rem;
            opacity: 0.8;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        }
        
        .card h2 {
            margin-bottom: 1rem;
            font-size: 1.4rem;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 0.5rem;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .status-item {
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        .status-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .status-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .freeze-risk {
            text-align: center;
            padding: 2rem;
        }
        
        .freeze-risk-gauge {
            width: 150px;
            height: 150px;
            margin: 0 auto 1rem;
            position: relative;
            border-radius: 50%;
            background: conic-gradient(from 0deg, #4caf50 0deg, #ffeb3b 90deg, #f44336 180deg);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .freeze-risk-inner {
            width: 120px;
            height: 120px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }
        
        .freeze-risk-value {
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        .freeze-risk-label {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .agents-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .agent-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        
        .agent-item.inactive {
            border-left-color: #f44336;
            opacity: 0.7;
        }
        
        .agent-item.error {
            border-left-color: #ff9800;
        }
        
        .agent-name {
            font-weight: bold;
            margin-bottom: 0.3rem;
        }
        
        .agent-model {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .agent-stats {
            text-align: right;
            font-size: 0.8rem;
        }
        
        .alerts-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .alert-item {
            padding: 1rem;
            margin-bottom: 0.8rem;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }
        
        .alert-item.warning {
            border-left-color: #ff9800;
            background: rgba(255, 152, 0, 0.1);
        }
        
        .alert-item.critical {
            border-left-color: #f44336;
            background: rgba(244, 67, 54, 0.1);
        }
        
        .alert-message {
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .alert-details {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .connection-status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .connection-status.connected {
            background: #4caf50;
        }
        
        .connection-status.disconnected {
            background: #f44336;
        }
        
        .metric-chart {
            height: 200px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin-top: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            opacity: 0.7;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
        
        .last-updated {
            text-align: center;
            margin-top: 2rem;
            font-size: 0.9rem;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 SutazAI Monitoring</h1>
        <div class="subtitle">Real-time monitoring for 131 Ollama-powered AI agents</div>
    </div>
    
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="dashboard">
        <!-- System Status Card -->
        <div class="card">
            <h2>📊 System Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="activeAgents">-</div>
                    <div class="status-label">Active Agents</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="totalAgents">-</div>
                    <div class="status-label">Total Agents</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="memoryUsage">-</div>
                    <div class="status-label">Memory Usage</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="cpuUsage">-</div>
                    <div class="status-label">CPU Usage</div>
                </div>
            </div>
        </div>
        
        <!-- Freeze Risk Card -->
        <div class="card">
            <h2>🚨 Freeze Risk Monitor</h2>
            <div class="freeze-risk">
                <div class="freeze-risk-gauge">
                    <div class="freeze-risk-inner">
                        <div class="freeze-risk-value" id="freezeRisk">-</div>
                        <div class="freeze-risk-label">Risk %</div>
                    </div>
                </div>
                <div id="freezeRiskStatus">Calculating...</div>
            </div>
        </div>
        
        <!-- Ollama Metrics Card -->
        <div class="card">
            <h2>🧠 Ollama Metrics</h2>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-value" id="queueDepth">-</div>
                    <div class="status-label">Queue Depth</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="activeConnections">-</div>
                    <div class="status-label">Active Connections</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="totalRequests">-</div>
                    <div class="status-label">Total Requests</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="avgResponseTime">-</div>
                    <div class="status-label">Avg Response (s)</div>
                </div>
            </div>
        </div>
        
        <!-- Active Agents Card -->
        <div class="card">
            <h2>🤖 Active Agents</h2>
            <div class="agents-list" id="agentsList">
                <div class="loading">Loading agents...</div>
            </div>
        </div>
        
        <!-- Recent Alerts Card -->
        <div class="card">
            <h2>⚠️ Recent Alerts</h2>
            <div class="alerts-list" id="alertsList">
                <div class="loading">Loading alerts...</div>
            </div>
        </div>
        
        <!-- Performance Trends Card -->
        <div class="card">
            <h2>📈 Performance Trends</h2>
            <div class="metric-chart">
                Chart visualization would go here
                <br>
                (Future enhancement: real-time charts)
            </div>
        </div>
    </div>
    
    <div class="last-updated" id="lastUpdated">Last updated: Never</div>
    
    <script>
        class OllamaMonitorDashboard {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 10;
                this.reconnectDelay = 5000;
                this.isConnected = false;
                
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.setupEventListeners();
                
                // Initial data fetch
                this.fetchInitialData();
                
                // Fallback polling in case WebSocket fails
                setInterval(() => {
                    if (!this.isConnected) {
                        this.fetchInitialData();
                    }
                }, 10000);
            }
            
            connectWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
                
                try {
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        console.log('WebSocket connected');
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus('connected');
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.updateDashboard(data);
                        } catch (error) {
                            console.error('Error parsing WebSocket message:', error);
                        }
                    };
                    
                    this.ws.onclose = () => {
                        console.log('WebSocket disconnected');
                        this.isConnected = false;
                        this.updateConnectionStatus('disconnected');
                        this.attemptReconnect();
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.isConnected = false;
                        this.updateConnectionStatus('disconnected');
                    };
                    
                } catch (error) {
                    console.error('Error creating WebSocket:', error);
                    this.attemptReconnect();
                }
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    
                    setTimeout(() => {
                        this.connectWebSocket();
                    }, this.reconnectDelay);
                } else {
                    console.error('Max reconnection attempts reached');
                    this.updateConnectionStatus('failed');
                }
            }
            
            updateConnectionStatus(status) {
                const statusElement = document.getElementById('connectionStatus');
                statusElement.className = `connection-status ${status}`;
                
                switch (status) {
                    case 'connected':
                        statusElement.textContent = '🟢 Connected';
                        break;
                    case 'disconnected':
                        statusElement.textContent = '🟡 Reconnecting...';
                        break;
                    case 'failed':
                        statusElement.textContent = '🔴 Connection Failed';
                        break;
                    default:
                        statusElement.textContent = '🟡 Connecting...';
                }
            }
            
            async fetchInitialData() {
                try {
                    const response = await fetch('/api/status');
                    if (response.ok) {
                        const data = await response.json();
                        this.updateDashboard(data);
                    }
                } catch (error) {
                    console.error('Error fetching initial data:', error);
                }
            }
            
            updateDashboard(data) {
                try {
                    // Update system status
                    if (data.system_status) {
                        document.getElementById('activeAgents').textContent = data.system_status.active_agents || '-';
                        document.getElementById('totalAgents').textContent = data.system_status.total_agents || '-';
                        document.getElementById('memoryUsage').textContent = 
                            data.system_status.memory_usage_percent ? 
                            `${data.system_status.memory_usage_percent.toFixed(1)}%` : '-';
                        document.getElementById('cpuUsage').textContent = 
                            data.system_status.cpu_usage_percent ? 
                            `${data.system_status.cpu_usage_percent.toFixed(1)}%` : '-';
                        
                        // Update freeze risk
                        const freezeRisk = data.system_status.freeze_risk_score || 0;
                        document.getElementById('freezeRisk').textContent = freezeRisk.toFixed(0);
                        
                        let riskStatus = 'Low Risk';
                        let riskColor = '#4caf50';
                        
                        if (freezeRisk > 80) {
                            riskStatus = 'CRITICAL RISK';
                            riskColor = '#f44336';
                        } else if (freezeRisk > 60) {
                            riskStatus = 'High Risk';
                            riskColor = '#ff9800';
                        } else if (freezeRisk > 40) {
                            riskStatus = 'Medium Risk';
                            riskColor = '#ffeb3b';
                        }
                        
                        document.getElementById('freezeRiskStatus').textContent = riskStatus;
                        document.getElementById('freezeRiskStatus').style.color = riskColor;
                    }
                    
                    // Update Ollama metrics
                    if (data.ollama_metrics) {
                        document.getElementById('queueDepth').textContent = data.ollama_metrics.queue_depth || '-';
                        document.getElementById('activeConnections').textContent = data.ollama_metrics.active_connections || '-';
                        document.getElementById('totalRequests').textContent = data.ollama_metrics.total_requests || '-';
                        document.getElementById('avgResponseTime').textContent = 
                            data.ollama_metrics.avg_response_time ? 
                            data.ollama_metrics.avg_response_time.toFixed(2) : '-';
                    }
                    
                    // Update agents list
                    if (data.agent_metrics) {
                        this.updateAgentsList(data.agent_metrics);
                    }
                    
                    // Update alerts
                    if (data.recent_alerts) {
                        this.updateAlertsList(data.recent_alerts);
                    }
                    
                    // Update timestamp
                    document.getElementById('lastUpdated').textContent = 
                        `Last updated: ${new Date().toLocaleTimeString()}`;
                    
                } catch (error) {
                    console.error('Error updating dashboard:', error);
                }
            }
            
            updateAgentsList(agentMetrics) {
                const agentsList = document.getElementById('agentsList');
                let html = '';
                
                const sortedAgents = Object.entries(agentMetrics)
                    .sort(([,a], [,b]) => {
                        // Sort by status first (active first), then by name
                        if (a.status !== b.status) {
                            return a.status === 'active' ? -1 : 1;
                        }
                        return a.agent_name.localeCompare(b.agent_name);
                    });
                
                for (const [name, metrics] of sortedAgents) {
                    const statusClass = metrics.status === 'active' ? '' : 
                                       metrics.status === 'error' ? 'error' : 'inactive';
                    
                    html += `
                        <div class="agent-item ${statusClass}">
                            <div>
                                <div class="agent-name">${metrics.agent_name}</div>
                                <div class="agent-model">Model: ${metrics.model}</div>
                            </div>
                            <div class="agent-stats">
                                <div>Tasks: ${metrics.tasks_processed}</div>
                                <div>Memory: ${metrics.memory_usage_mb.toFixed(0)}MB</div>
                                <div>Ollama: ${metrics.ollama_requests}</div>
                            </div>
                        </div>
                    `;
                }
                
                agentsList.innerHTML = html || '<div class="loading">No agents found</div>';
            }
            
            updateAlertsList(alerts) {
                const alertsList = document.getElementById('alertsList');
                let html = '';
                
                if (alerts.length === 0) {
                    html = '<div style="text-align: center; opacity: 0.6;">🎉 No active alerts</div>';
                } else {
                    for (const alert of alerts.slice(0, 10)) {  // Show only latest 10
                        const severityClass = alert.severity || 'warning';
                        const timestamp = new Date(alert.timestamp).toLocaleTimeString();
                        
                        html += `
                            <div class="alert-item ${severityClass}">
                                <div class="alert-message">${alert.message}</div>
                                <div class="alert-details">
                                    Type: ${alert.alert_type} | Agent: ${alert.agent_name || 'System'} | ${timestamp}
                                </div>
                            </div>
                        `;
                    }
                }
                
                alertsList.innerHTML = html;
            }
            
            setupEventListeners() {
                // Handle page visibility changes
                document.addEventListener('visibilitychange', () => {
                    if (document.hidden) {
                        // Page is hidden, reduce update frequency
                        console.log('Page hidden, reducing updates');
                    } else {
                        // Page is visible again, resume normal updates
                        console.log('Page visible, resuming updates');
                        if (this.isConnected) {
                            this.fetchInitialData();
                        }
                    }
                });
                
                // Handle window focus/blur
                window.addEventListener('focus', () => {
                    if (this.isConnected) {
                        this.fetchInitialData();
                    }
                });
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new OllamaMonitorDashboard();
        });
    </script>
</body>
</html>
        """
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'ollama-realtime-dashboard'
        })
    
    async def get_system_status(self, request):
        """Get current system status"""
        try:
            dashboard_data = await self.monitor.get_dashboard_data()
            return web.json_response(dashboard_data)
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_agents_status(self, request):
        """Get detailed agent status"""
        try:
            agents_data = {name: metrics for name, metrics in self.monitor.known_agents.items()}
            return web.json_response({
                'agents': agents_data,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting agents status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_alerts(self, request):
        """Get recent alerts"""
        try:
            with sqlite3.connect(self.monitor.db_path) as conn:
                cursor = conn.execute('''
                    SELECT alert_type, severity, message, agent_name, metric_value, threshold, timestamp
                    FROM alerts
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''')
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'alert_type': row[0],
                        'severity': row[1],
                        'message': row[2],
                        'agent_name': row[3],
                        'metric_value': row[4],
                        'threshold': row[5],
                        'timestamp': row[6]
                    })
                
                return web.json_response({
                    'alerts': alerts,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_metrics_history(self, request):
        """Get historical metrics data"""
        try:
            hours = int(request.query.get('hours', 1))
            since = datetime.utcnow() - timedelta(hours=hours)
            
            with sqlite3.connect(self.monitor.db_path) as conn:
                # System metrics
                cursor = conn.execute('''
                    SELECT timestamp, total_requests, total_failures, queue_depth, 
                           avg_response_time, memory_usage_mb, cpu_usage_percent
                    FROM system_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp
                ''', (since.isoformat(),))
                
                system_metrics = []
                for row in cursor.fetchall():
                    system_metrics.append({
                        'timestamp': row[0],
                        'total_requests': row[1],
                        'total_failures': row[2],
                        'queue_depth': row[3],
                        'avg_response_time': row[4],
                        'memory_usage_mb': row[5],
                        'cpu_usage_percent': row[6]
                    })
                
                return web.json_response({
                    'system_metrics': system_metrics,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        logger.info("New WebSocket connection established")
        
        try:
            # Send initial data
            dashboard_data = await self.monitor.get_dashboard_data()
            await ws.send_str(json.dumps(dashboard_data))
            
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # Handle client messages if needed
                    except json.JSONDecodeError:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.websocket_connections.discard(ws)
            logger.info("WebSocket connection closed")
        
        return ws
    
    async def broadcast_updates(self):
        """Broadcast updates to all connected WebSocket clients"""
        while not self.shutdown_event.is_set():
            try:
                if self.websocket_connections:
                    dashboard_data = await self.monitor.get_dashboard_data()
                    message = json.dumps(dashboard_data)
                    
                    # Send to all connected clients
                    disconnected = set()
                    for ws in self.websocket_connections:
                        try:
                            await ws.send_str(message)
                        except Exception as e:
                            logger.debug(f"Error sending to WebSocket: {e}")
                            disconnected.add(ws)
                    
                    # Remove disconnected clients
                    self.websocket_connections -= disconnected
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(10)
    
    async def start(self):
        """Start the dashboard server"""
        logger.info(f"Starting real-time dashboard on port {self.port}")
        
        # Start the monitor
        monitor_task = asyncio.create_task(self.monitor.start())
        
        # Start WebSocket broadcasting
        broadcast_task = asyncio.create_task(self.broadcast_updates())
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Dashboard available at http://localhost:{self.port}")
        
        try:
            await asyncio.gather(monitor_task, broadcast_task)
        except asyncio.CancelledError:
            logger.info("Dashboard tasks cancelled")
        finally:
            await runner.cleanup()
            await self.monitor.stop()
    
    def stop(self):
        """Stop the dashboard"""
        self.shutdown_event.set()


async def main():
    """Main entry point"""
    dashboard = RealtimeDashboard()
    
    # Setup signal handlers
    def signal_handler():
        logger.info("Received shutdown signal")
        dashboard.stop()
    
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        await dashboard.start()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())