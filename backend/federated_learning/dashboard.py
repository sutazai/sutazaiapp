"""
Federated Learning Dashboard
============================

Web-based dashboard for monitoring and managing federated learning in SutazAI.
Provides real-time visualization, control interface, and analytics for federated training.

Features:
- Real-time training progress visualization
- Client performance monitoring
- Privacy budget tracking
- Model performance analytics
- Training control interface
- Alert management
- System health monitoring
- Interactive charts and graphs
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .coordinator import FederatedCoordinator, get_federated_coordinator, TrainingConfiguration, ClientSelectionStrategy
from .aggregator import AggregationAlgorithm
from .monitoring import FederatedMonitor, MetricType, AlertSeverity, TrainingMetric
from .privacy import PrivacyLevel, PrivacyBudget, PrivacyMechanism
from .versioning import ModelVersionManager


# Pydantic models for API
class TrainingRequest(BaseModel):
    name: str
    algorithm: str
    model_type: str
    target_accuracy: float
    max_rounds: int
    min_clients_per_round: int
    max_clients_per_round: int
    client_selection_strategy: str
    local_epochs: int
    local_batch_size: int
    local_learning_rate: float
    privacy_level: Optional[str] = None


class MetricRequest(BaseModel):
    training_id: str
    round_number: int
    metric_type: str
    value: float
    client_id: Optional[str] = None


class ClientPerformanceRequest(BaseModel):
    client_id: str
    training_id: str
    training_time: float
    communication_time: float
    samples_contributed: int
    success: bool


class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_subscriptions: Dict[str, List[str]] = {}  # client_id -> [training_ids]
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_subscriptions[client_id] = []
        
    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
    
    async def subscribe_to_training(self, client_id: str, training_id: str):
        if client_id in self.client_subscriptions:
            if training_id not in self.client_subscriptions[client_id]:
                self.client_subscriptions[client_id].append(training_id)
    
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)
    
    async def send_to_subscribers(self, training_id: str, message: Dict[str, Any]):
        """Send message to clients subscribed to specific training"""
        if self.active_connections:
            disconnected = []
            for i, connection in enumerate(self.active_connections):
                # Check if this client is subscribed to the training
                client_id = list(self.client_subscriptions.keys())[i] if i < len(self.client_subscriptions) else None
                
                if client_id and training_id in self.client_subscriptions.get(client_id, []):
                    try:
                        await connection.send_json(message)
                    except:
                        disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)


class FederatedDashboard:
    """
    Federated Learning Dashboard
    
    Web-based interface for monitoring and controlling federated learning
    in the SutazAI system.
    """
    
    def __init__(self, 
                 coordinator: FederatedCoordinator = None,
                 monitor: FederatedMonitor = None,
                 version_manager: ModelVersionManager = None,
                 host: str = "0.0.0.0",
                 port: int = 8000):
        
        self.coordinator = coordinator
        self.monitor = monitor
        self.version_manager = version_manager
        self.host = host
        self.port = port
        
        # WebSocket manager
        self.websocket_manager = WebSocketManager()
        
        # FastAPI app
        self.app = FastAPI(
            title="SutazAI Federated Learning Dashboard",
            description="Monitor and control federated learning across AI agents",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("federated_dashboard")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Main dashboard page
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._generate_dashboard_html()
        
        # API Routes
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Training management
        @self.app.post("/api/training/start")
        async def start_training(request: TrainingRequest):
            try:
                if not self.coordinator:
                    raise HTTPException(status_code=503, detail="Coordinator not available")
                
                # Convert request to training configuration
                config = self._create_training_config(request)
                
                # Start training
                training_id = await self.coordinator.start_training(config)
                
                # Broadcast update
                await self.websocket_manager.broadcast_update({
                    "type": "training_started",
                    "training_id": training_id,
                    "config": request.dict()
                })
                
                return {"training_id": training_id, "status": "started"}
                
            except Exception as e:
                self.logger.error(f"Failed to start training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/training/{training_id}/stop")
        async def stop_training(training_id: str):
            try:
                if not self.coordinator:
                    raise HTTPException(status_code=503, detail="Coordinator not available")
                
                success = await self.coordinator.stop_training(training_id)
                
                if success:
                    await self.websocket_manager.send_to_subscribers(training_id, {
                        "type": "training_stopped",
                        "training_id": training_id
                    })
                    return {"status": "stopped"}
                else:
                    raise HTTPException(status_code=404, detail="Training not found")
                
            except Exception as e:
                self.logger.error(f"Failed to stop training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/{training_id}/status")
        async def get_training_status(training_id: str):
            try:
                if not self.coordinator:
                    raise HTTPException(status_code=503, detail="Coordinator not available")
                
                status = self.coordinator.get_training_status(training_id)
                if not status:
                    raise HTTPException(status_code=404, detail="Training not found")
                
                return status
                
            except Exception as e:
                self.logger.error(f"Failed to get training status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training")
        async def list_trainings():
            try:
                if not self.coordinator:
                    raise HTTPException(status_code=503, detail="Coordinator not available")
                
                active_trainings = self.coordinator.get_active_trainings()
                training_statuses = []
                
                for training_id in active_trainings:
                    status = self.coordinator.get_training_status(training_id)
                    if status:
                        training_statuses.append(status)
                
                return {"trainings": training_statuses}
                
            except Exception as e:
                self.logger.error(f"Failed to list trainings: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Monitoring endpoints
        @self.app.post("/api/metrics")
        async def record_metric(request: MetricRequest):
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                metric = TrainingMetric(
                    training_id=request.training_id,
                    round_number=request.round_number,
                    metric_type=MetricType(request.metric_type),
                    value=request.value,
                    timestamp=datetime.utcnow(),
                    client_id=request.client_id
                )
                
                await self.monitor.record_training_metric(metric)
                
                # Broadcast metric update
                await self.websocket_manager.send_to_subscribers(request.training_id, {
                    "type": "metric_update",
                    "training_id": request.training_id,
                    "metric": metric.to_dict()
                })
                
                return {"status": "recorded"}
                
            except Exception as e:
                self.logger.error(f"Failed to record metric: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/{training_id}/metrics")
        async def get_training_metrics(training_id: str, metric_type: Optional[str] = None):
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                metric_type_enum = MetricType(metric_type) if metric_type else None
                metrics = self.monitor.get_training_metrics(training_id, metric_type_enum)
                
                return {"metrics": metrics}
                
            except Exception as e:
                self.logger.error(f"Failed to get training metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/{training_id}/progress")
        async def get_training_progress(training_id: str):
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                progress = self.monitor.get_training_progress(training_id)
                if not progress:
                    raise HTTPException(status_code=404, detail="Training progress not found")
                
                return progress
                
            except Exception as e:
                self.logger.error(f"Failed to get training progress: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/client/performance")
        async def record_client_performance(request: ClientPerformanceRequest):
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                await self.monitor.record_client_performance(
                    request.client_id,
                    request.training_id,
                    request.training_time,
                    request.communication_time,
                    request.samples_contributed,
                    request.success
                )
                
                return {"status": "recorded"}
                
            except Exception as e:
                self.logger.error(f"Failed to record client performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/client/{client_id}/performance")
        async def get_client_performance(client_id: str):
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                performance = self.monitor.get_client_performance(client_id)
                if not performance:
                    raise HTTPException(status_code=404, detail="Client performance not found")
                
                return performance
                
            except Exception as e:
                self.logger.error(f"Failed to get client performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # System monitoring
        @self.app.get("/api/system/health")
        async def get_system_health():
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                health = self.monitor.get_system_health()
                return health
                
            except Exception as e:
                self.logger.error(f"Failed to get system health: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts")
        async def get_alerts(training_id: Optional[str] = None, severity: Optional[str] = None):
            try:
                if not self.monitor:
                    raise HTTPException(status_code=503, detail="Monitor not available")
                
                severity_enum = AlertSeverity(severity) if severity else None
                alerts = self.monitor.get_alerts(training_id, severity_enum)
                
                return {"alerts": alerts}
                
            except Exception as e:
                self.logger.error(f"Failed to get alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Model versioning
        @self.app.get("/api/training/{training_id}/versions")
        async def get_model_versions(training_id: str):
            try:
                if not self.version_manager:
                    raise HTTPException(status_code=503, detail="Version manager not available")
                
                versions = self.version_manager.get_version_history(training_id)
                active_version = self.version_manager.get_active_version(training_id)
                best_version = self.version_manager.get_best_version(training_id)
                
                return {
                    "versions": versions,
                    "active_version": active_version,
                    "best_version": best_version
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get model versions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/training/{training_id}/rollback")
        async def rollback_training(training_id: str, target_version: Optional[str] = None):
            try:
                if not self.version_manager:
                    raise HTTPException(status_code=503, detail="Version manager not available")
                
                rollback_version = await self.version_manager.rollback_to_version(training_id, target_version)
                
                if rollback_version:
                    await self.websocket_manager.send_to_subscribers(training_id, {
                        "type": "model_rollback",
                        "training_id": training_id,
                        "rollback_version": rollback_version
                    })
                    return {"rollback_version": rollback_version}
                else:
                    raise HTTPException(status_code=400, detail="Rollback failed")
                
            except Exception as e:
                self.logger.error(f"Failed to rollback training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Statistics and analytics
        @self.app.get("/api/stats/coordinator")
        async def get_coordinator_stats():
            try:
                if not self.coordinator:
                    raise HTTPException(status_code=503, detail="Coordinator not available")
                
                stats = self.coordinator.get_coordinator_stats()
                return stats
                
            except Exception as e:
                self.logger.error(f"Failed to get coordinator stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/stats/versioning")
        async def get_versioning_stats():
            try:
                if not self.version_manager:
                    raise HTTPException(status_code=503, detail="Version manager not available")
                
                stats = self.version_manager.get_version_stats()
                return stats
                
            except Exception as e:
                self.logger.error(f"Failed to get versioning stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.websocket_manager.connect(websocket, client_id)
            try:
                while True:
                    data = await websocket.receive_json()
                    
                    # Handle subscription requests
                    if data.get("action") == "subscribe":
                        training_id = data.get("training_id")
                        if training_id:
                            await self.websocket_manager.subscribe_to_training(client_id, training_id)
                            await websocket.send_json({
                                "type": "subscription_confirmed",
                                "training_id": training_id
                            })
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket, client_id)
    
    def _create_training_config(self, request: TrainingRequest) -> TrainingConfiguration:
        """Convert API request to training configuration"""
        
        # Create privacy budget if specified
        privacy_budget = None
        if request.privacy_level:
            privacy_level = PrivacyLevel(request.privacy_level)
            privacy_budget = PrivacyBudget(
                total_epsilon=10.0 if privacy_level == PrivacyLevel.LOW else 1.0,
                total_delta=1e-5,
                mechanism=PrivacyMechanism.GAUSSIAN_DP
            )
        
        config = TrainingConfiguration(
            name=request.name,
            algorithm=AggregationAlgorithm(request.algorithm),
            model_type=request.model_type,
            target_accuracy=request.target_accuracy,
            max_rounds=request.max_rounds,
            min_clients_per_round=request.min_clients_per_round,
            max_clients_per_round=request.max_clients_per_round,
            client_selection_strategy=ClientSelectionStrategy(request.client_selection_strategy),
            local_epochs=request.local_epochs,
            local_batch_size=request.local_batch_size,
            local_learning_rate=request.local_learning_rate,
            convergence_threshold=0.01,
            privacy_budget=privacy_budget,
            timeout_seconds=1800,
            validation_frequency=5,
            early_stopping_patience=10,
            resource_constraints={"max_memory_mb": 2048, "max_cpu_percent": 80}
        )
        
        return config
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML page"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Federated Learning Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin: 0;
            font-size: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .status-healthy { background: #d4edda; color: #155724; }
        .status-warning { background: #fff3cd; color: #856404; }
        .status-error { background: #f8d7da; color: #721c24; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-value {
            font-weight: bold;
            color: #667eea;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .controls {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: #333;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
        }
        
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            margin-right: 0.5rem;
        }
        
        .btn:hover {
            background: #5a6fd8;
        }
        
        .btn-danger {
            background: #dc3545;
        }
        
        .btn-danger:hover {
            background: #c82333;
        }
        
        .alert {
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .alert-info { background: #d1ecf1; color: #0c5460; }
        .alert-warning { background: #fff3cd; color: #856404; }
        .alert-error { background: #f8d7da; color: #721c24; }
        
        #connectionStatus {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }
        
        .connected { background: #28a745; }
        .disconnected { background: #dc3545; }
    </style>
</head>
<body>
    <div id="connectionStatus" class="disconnected">Disconnected</div>
    
    <div class="header">
        <h1>🤖 SutazAI Federated Learning Dashboard</h1>
        <p>Monitor and control distributed AI training across 69 agents</p>
    </div>
    
    <div class="container">
        <!-- Training Controls -->
        <div class="controls">
            <h2>🚀 Start New Training</h2>
            <form id="trainingForm">
                <div class="grid">
                    <div class="form-group">
                        <label for="trainingName">Training Name</label>
                        <input type="text" id="trainingName" name="name" required>
                    </div>
                    <div class="form-group">
                        <label for="algorithm">Algorithm</label>
                        <select id="algorithm" name="algorithm" required>
                            <option value="fedavg">FedAvg</option>
                            <option value="fedprox">FedProx</option>
                            <option value="fedopt">FedOpt</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="modelType">Model Type</label>
                        <select id="modelType" name="model_type" required>
                            <option value="neural_network">Neural Network</option>
                            <option value="linear_regression">Linear Regression</option>
                            <option value="logistic_regression">Logistic Regression</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="targetAccuracy">Target Accuracy</label>
                        <input type="number" id="targetAccuracy" name="target_accuracy" 
                               min="0" max="1" step="0.01" value="0.95" required>
                    </div>
                    <div class="form-group">
                        <label for="maxRounds">Maximum Rounds</label>
                        <input type="number" id="maxRounds" name="max_rounds" 
                               min="1" max="1000" value="100" required>
                    </div>
                    <div class="form-group">
                        <label for="minClients">Minimum Clients per Round</label>
                        <input type="number" id="minClients" name="min_clients_per_round" 
                               min="1" max="69" value="5" required>
                    </div>
                </div>
                <button type="submit" class="btn">Start Training</button>
            </form>
        </div>
        
        <!-- System Health -->
        <div class="card">
            <h2>🏥 System Health</h2>
            <div id="systemHealth">
                <div class="metric">
                    <span>Overall Health</span>
                    <span class="metric-value">Loading...</span>
                </div>
                <div class="metric">
                    <span>Active Trainings</span>
                    <span class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span>Connected Clients</span>
                    <span class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span>Active Alerts</span>
                    <span class="metric-value">0</span>
                </div>
            </div>
        </div>
        
        <!-- Active Trainings -->
        <div class="card">
            <h2>🎯 Active Trainings</h2>
            <div id="activeTrainings">
                <p>No active trainings</p>
            </div>
        </div>
        
        <div class="grid">
            <!-- Training Metrics Chart -->
            <div class="card">
                <h2>📊 Training Metrics</h2>
                <div class="chart-container">
                    <canvas id="metricsChart"></canvas>
                </div>
            </div>
            
            <!-- Client Performance -->
            <div class="card">
                <h2>👥 Client Performance</h2>
                <div class="chart-container">
                    <canvas id="clientChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Recent Alerts -->
        <div class="card">
            <h2>🚨 Recent Alerts</h2>
            <div id="recentAlerts">
                <p>No recent alerts</p>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws = null;
        let reconnectTimer = null;
        
        // Charts
        let metricsChart = null;
        let clientChart = null;
        
        // Data
        let trainingData = {};
        let systemHealthData = {};
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            setupCharts();
            setupEventListeners();
            loadInitialData();
            
            // Refresh data periodically
            setInterval(refreshData, 30000); // Every 30 seconds
        });
        
        function connectWebSocket() {
            const clientId = 'dashboard_' + Math.random().toString(36).substr(2, 9);
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                updateConnectionStatus(true);
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };
            
            ws.onclose = function() {
                updateConnectionStatus(false);
                console.log('WebSocket disconnected');
                
                // Attempt to reconnect
                reconnectTimer = setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateConnectionStatus(connected) {
            const statusElement = document.getElementById('connectionStatus');
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = connected ? 'connected' : 'disconnected';
        }
        
        function handleWebSocketMessage(message) {
            console.log('Received message:', message);
            
            switch(message.type) {
                case 'training_started':
                    refreshActiveTrainings();
                    break;
                case 'training_stopped':
                    refreshActiveTrainings();
                    break;
                case 'metric_update':
                    updateMetricsChart(message.metric);
                    break;
                case 'model_rollback':
                    showAlert('Model rollback completed', 'info');
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        }
        
        function setupCharts() {
            // Metrics chart
            const metricsCtx = document.getElementById('metricsChart').getContext('2d');
            metricsChart = new Chart(metricsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Accuracy',
                        data: [],
                        borderColor: 'rgb(102, 126, 234)',
                        tension: 0.1
                    }, {
                        label: 'Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Client performance chart
            const clientCtx = document.getElementById('clientChart').getContext('2d');
            clientChart = new Chart(clientCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Reliability Score',
                        data: [],
                        backgroundColor: 'rgba(102, 126, 234, 0.5)',
                        borderColor: 'rgb(102, 126, 234)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }
        
        function setupEventListeners() {
            // Training form
            document.getElementById('trainingForm').addEventListener('submit', function(e) {
                e.preventDefault();
                startTraining();
            });
        }
        
        async function startTraining() {
            const form = document.getElementById('trainingForm');
            const formData = new FormData(form);
            
            const trainingRequest = {
                name: formData.get('name'),
                algorithm: formData.get('algorithm'),
                model_type: formData.get('model_type'),
                target_accuracy: parseFloat(formData.get('target_accuracy')),
                max_rounds: parseInt(formData.get('max_rounds')),
                min_clients_per_round: parseInt(formData.get('min_clients_per_round')),
                max_clients_per_round: 20,
                client_selection_strategy: 'random',
                local_epochs: 5,
                local_batch_size: 32,
                local_learning_rate: 0.01
            };
            
            try {
                const response = await fetch('/api/training/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(trainingRequest)
                });
                
                if (response.ok) {
                    const result = await response.json();
                    showAlert(`Training started: ${result.training_id}`, 'info');
                    form.reset();
                } else {
                    const error = await response.json();
                    showAlert(`Failed to start training: ${error.detail}`, 'error');
                }
            } catch (error) {
                showAlert(`Error: ${error.message}`, 'error');
            }
        }
        
        async function loadInitialData() {
            await Promise.all([
                refreshSystemHealth(),
                refreshActiveTrainings(),
                refreshAlerts()
            ]);
        }
        
        async function refreshData() {
            await Promise.all([
                refreshSystemHealth(),
                refreshActiveTrainings()
            ]);
        }
        
        async function refreshSystemHealth() {
            try {
                const response = await fetch('/api/system/health');
                if (response.ok) {
                    systemHealthData = await response.json();
                    updateSystemHealthDisplay();
                }
            } catch (error) {
                console.error('Error fetching system health:', error);
            }
        }
        
        function updateSystemHealthDisplay() {
            const healthElement = document.getElementById('systemHealth');
            const health = systemHealthData;
            
            if (!health) return;
            
            const statusClass = health.health_status === 'healthy' ? 'status-healthy' : 
                               health.health_status === 'warning' ? 'status-warning' : 'status-error';
            
            healthElement.innerHTML = `
                <div class="metric">
                    <span>Overall Health</span>
                    <span class="status-badge ${statusClass}">${health.health_status}</span>
                </div>
                <div class="metric">
                    <span>Health Score</span>
                    <span class="metric-value">${(health.overall_health_score * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span>Active Trainings</span>
                    <span class="metric-value">${health.training_summary?.total_trainings || 0}</span>
                </div>
                <div class="metric">
                    <span>Active Clients</span>
                    <span class="metric-value">${health.training_summary?.active_clients || 0}</span>
                </div>
                <div class="metric">
                    <span>Active Alerts</span>
                    <span class="metric-value">${health.active_alerts || 0}</span>
                </div>
            `;
        }
        
        async function refreshActiveTrainings() {
            try {
                const response = await fetch('/api/training');
                if (response.ok) {
                    const data = await response.json();
                    updateActiveTrainingsDisplay(data.trainings);
                }
            } catch (error) {
                console.error('Error fetching active trainings:', error);
            }
        }
        
        function updateActiveTrainingsDisplay(trainings) {
            const container = document.getElementById('activeTrainings');
            
            if (!trainings || trainings.length === 0) {
                container.innerHTML = '<p>No active trainings</p>';
                return;
            }
            
            container.innerHTML = trainings.map(training => `
                <div class="card" style="margin-bottom: 1rem;">
                    <h3>${training.training_id}</h3>
                    <div class="metric">
                        <span>Status</span>
                        <span class="status-badge status-${training.status === 'active' ? 'healthy' : 'warning'}">
                            ${training.status}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Round</span>
                        <span class="metric-value">${training.current_round}</span>
                    </div>
                    <div class="metric">
                        <span>Clients</span>
                        <span class="metric-value">${training.participating_clients}</span>
                    </div>
                    ${training.latest_performance ? `
                        <div class="metric">
                            <span>Latest Accuracy</span>
                            <span class="metric-value">${(training.latest_performance.accuracy * 100).toFixed(2)}%</span>
                        </div>
                    ` : ''}
                    <button class="btn btn-danger" onclick="stopTraining('${training.training_id}')">
                        Stop Training
                    </button>
                </div>
            `).join('');
        }
        
        async function stopTraining(trainingId) {
            if (!confirm('Are you sure you want to stop this training?')) return;
            
            try {
                const response = await fetch(`/api/training/${trainingId}/stop`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    showAlert('Training stopped', 'info');
                } else {
                    showAlert('Failed to stop training', 'error');
                }
            } catch (error) {
                showAlert(`Error: ${error.message}`, 'error');
            }
        }
        
        async function refreshAlerts() {
            try {
                const response = await fetch('/api/alerts');
                if (response.ok) {
                    const data = await response.json();
                    updateAlertsDisplay(data.alerts);
                }
            } catch (error) {
                console.error('Error fetching alerts:', error);
            }
        }
        
        function updateAlertsDisplay(alerts) {
            const container = document.getElementById('recentAlerts');
            
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<p>No recent alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.slice(0, 5).map(alert => `
                <div class="alert alert-${alert.severity}">
                    <strong>${alert.title}</strong><br>
                    ${alert.description}<br>
                    <small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }
        
        function updateMetricsChart(metric) {
            if (!metricsChart) return;
            
            const roundLabel = `Round ${metric.round_number}`;
            
            // Add new label if needed
            if (!metricsChart.data.labels.includes(roundLabel)) {
                metricsChart.data.labels.push(roundLabel);
            }
            
            // Update dataset based on metric type
            if (metric.metric_type === 'accuracy') {
                metricsChart.data.datasets[0].data.push(metric.value);
            } else if (metric.metric_type === 'loss') {
                metricsChart.data.datasets[1].data.push(metric.value);
            }
            
            // Keep only last 20 points
            if (metricsChart.data.labels.length > 20) {
                metricsChart.data.labels.shift();
                metricsChart.data.datasets.forEach(dataset => {
                    if (dataset.data.length > 20) {
                        dataset.data.shift();
                    }
                });
            }
            
            metricsChart.update();
        }
        
        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type}`;
            alertDiv.textContent = message;
            alertDiv.style.position = 'fixed';
            alertDiv.style.top = '100px';
            alertDiv.style.right = '20px';
            alertDiv.style.zIndex = '1000';
            alertDiv.style.maxWidth = '300px';
            
            document.body.appendChild(alertDiv);
            
            setTimeout(() => {
                document.body.removeChild(alertDiv);
            }, 5000);
        }
    </script>
</body>
</html>
        '''
    
    async def start_server(self):
        """Start the dashboard server"""
        try:
            self.logger.info(f"Starting Federated Learning Dashboard on {self.host}:{self.port}")
            
            # Start background tasks
            self._start_background_tasks()
            
            # Start server
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background tasks"""
        tasks = [
            self._periodic_updates()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _periodic_updates(self):
        """Send periodic updates to connected clients"""
        while not self._shutdown_event.is_set():
            try:
                # Send system health updates
                if self.monitor:
                    health = self.monitor.get_system_health()
                    await self.websocket_manager.broadcast_update({
                        "type": "system_health_update",
                        "health": health
                    })
                
                await asyncio.sleep(30)  # Send updates every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Periodic updates error: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Shutdown the dashboard"""
        self.logger.info("Shutting down Federated Learning Dashboard")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Federated Learning Dashboard shutdown complete")


# Standalone server function
async def run_dashboard(coordinator: FederatedCoordinator = None,
                       monitor: FederatedMonitor = None,
                       version_manager: ModelVersionManager = None,
                       host: str = "0.0.0.0",
                       port: int = 8000):
    """Run the federated learning dashboard"""
    dashboard = FederatedDashboard(
        coordinator=coordinator,
        monitor=monitor,
        version_manager=version_manager,
        host=host,
        port=port
    )
    
    await dashboard.start_server()


if __name__ == "__main__":
    asyncio.run(run_dashboard())