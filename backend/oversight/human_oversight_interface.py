#!/usr/bin/env python3
"""
Human Oversight Interface for SutazAI System
Provides comprehensive human control and oversight for 69 AI agents
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class OverrideType(Enum):
    PARAMETER_CHANGE = "parameter_change"
    DECISION_OVERRIDE = "decision_override"
    POLICY_ENFORCEMENT = "policy_enforcement"
    EMERGENCY_STOP = "emergency_stop"
    PAUSE_AGENT = "pause_agent"
    RESUME_AGENT = "resume_agent"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class HumanOverride:
    """Represents a human intervention in AI system operations"""
    id: str
    agent_id: str
    override_type: OverrideType
    description: str
    previous_value: Any
    new_value: Any
    operator_id: str
    timestamp: datetime
    is_active: bool = True
    expiry_time: Optional[datetime] = None

@dataclass
class ApprovalRequest:
    """Represents a request requiring human approval"""
    id: str
    agent_id: str
    request_type: str
    description: str
    context: Dict[str, Any]
    risk_level: AlertSeverity
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver_id: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    approval_note: Optional[str] = None

@dataclass
class AuditEvent:
    """Represents an audit event for compliance tracking"""
    id: str
    event_type: str
    agent_id: str
    operator_id: str
    description: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    timestamp: datetime
    compliance_tags: List[str]

class HumanOversightInterface:
    """
    Comprehensive human oversight interface for SutazAI system
    Provides real-time monitoring, control, and intervention capabilities
    """
    
    def __init__(self, port: int = 8095, db_path: str = "/opt/sutazaiapp/backend/oversight/oversight.db"):
        self.port = port
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.app = web.Application()
        self.websocket_connections: set = set()
        self.shutdown_event = asyncio.Event()
        
        # Agent registry and status tracking
        self.agent_registry = {}
        self.agent_status = {}
        self.active_overrides = {}
        self.pending_approvals = {}
        
        # Initialize database
        self._init_database()
        
        # Load agent data
        self._load_agent_data()
        
        # Setup routes
        self._setup_routes()
        
        # CORS setup
        self._setup_cors()
    
    def _init_database(self):
        """Initialize SQLite database for oversight data"""
        with sqlite3.connect(self.db_path) as conn:
            # Human overrides table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS human_overrides (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    override_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    previous_value TEXT,
                    new_value TEXT,
                    operator_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    expiry_time TEXT
                )
            ''')
            
            # Approval requests table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    context TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    approver_id TEXT,
                    approval_timestamp TEXT,
                    approval_note TEXT
                )
            ''')
            
            # Audit events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    before_state TEXT,
                    after_state TEXT,
                    timestamp TEXT NOT NULL,
                    compliance_tags TEXT
                )
            ''')
            
            # Alert notifications table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_notifications (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    agent_id TEXT,
                    created_at TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            ''')
            
            # Agent control states table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_control_states (
                    agent_id TEXT PRIMARY KEY,
                    is_paused INTEGER DEFAULT 0,
                    paused_by TEXT,
                    paused_at TEXT,
                    pause_reason TEXT,
                    emergency_stopped INTEGER DEFAULT 0,
                    stopped_by TEXT,
                    stopped_at TEXT,
                    stop_reason TEXT,
                    parameter_overrides TEXT,
                    last_update TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def _load_agent_data(self):
        """Load agent registry and status data"""
        try:
            # Load agent registry
            registry_path = Path("/opt/sutazaiapp/agents/agent_registry.json")
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    self.agent_registry = data.get('agents', {})
            
            # Load agent status
            status_path = Path("/opt/sutazaiapp/agents/agent_status.json")
            if status_path.exists():
                with open(status_path, 'r') as f:
                    data = json.load(f)
                    self.agent_status = data.get('active_agents', {})
                    
            logger.info(f"Loaded {len(self.agent_registry)} agents from registry")
            logger.info(f"Loaded {len(self.agent_status)} active agents from status")
                    
        except Exception as e:
            logger.error(f"Error loading agent data: {e}")
    
    def _setup_routes(self):
        """Setup HTTP and WebSocket routes"""
        
        # Main dashboard
        self.app.router.add_get('/', self.serve_oversight_dashboard)
        self.app.router.add_get('/health', self.health_check)
        
        # API endpoints for real-time data
        self.app.router.add_get('/api/agents/status', self.get_agents_status)
        self.app.router.add_get('/api/agents/metrics', self.get_agents_metrics)
        self.app.router.add_get('/api/overrides', self.get_active_overrides)
        self.app.router.add_get('/api/approvals', self.get_pending_approvals)
        self.app.router.add_get('/api/alerts', self.get_active_alerts)
        self.app.router.add_get('/api/audit', self.get_audit_events)
        
        # Control endpoints
        self.app.router.add_post('/api/agents/{agent_id}/pause', self.pause_agent)
        self.app.router.add_post('/api/agents/{agent_id}/resume', self.resume_agent)
        self.app.router.add_post('/api/agents/{agent_id}/emergency_stop', self.emergency_stop_agent)
        self.app.router.add_post('/api/agents/{agent_id}/override', self.create_override)
        self.app.router.add_post('/api/approvals/{approval_id}/approve', self.approve_request)
        self.app.router.add_post('/api/approvals/{approval_id}/reject', self.reject_request)
        self.app.router.add_post('/api/alerts/{alert_id}/acknowledge', self.acknowledge_alert)
        
        # Bulk operations
        self.app.router.add_post('/api/agents/bulk/pause', self.bulk_pause_agents)
        self.app.router.add_post('/api/agents/bulk/resume', self.bulk_resume_agents)
        self.app.router.add_post('/api/system/emergency_stop', self.system_emergency_stop)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self.websocket_handler)
    
    def _setup_cors(self):
        """Setup CORS configuration"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="Content-Type, Content-Length, Authorization, Cache-Control, Expires",
                allow_headers="Accept, Accept-Language, Content-Type, Content-Language, Authorization, X-Requested-With, X-CSRFToken, Cache-Control",
                allow_methods="GET, POST, PUT, DELETE, OPTIONS, PATCH"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def serve_oversight_dashboard(self, request):
        """Serve the main oversight dashboard HTML"""
        html_content = self._generate_oversight_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    def _generate_oversight_dashboard_html(self) -> str:
        """Generate comprehensive oversight dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI - Human Oversight Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        
        .emergency-controls {
            display: flex;
            gap: 1rem;
        }
        
        .emergency-btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .emergency-stop {
            background: #f44336;
            color: white;
        }
        
        .emergency-stop:hover {
            background: #d32f2f;
            transform: scale(1.05);
        }
        
        .bulk-pause {
            background: #ff9800;
            color: white;
        }
        
        .bulk-pause:hover {
            background: #f57c00;
            transform: scale(1.05);
        }
        
        .tabs {
            display: flex;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .tab {
            padding: 1rem 2rem;
            cursor: pointer;
            transition: background 0.3s ease;
            border-bottom: 3px solid transparent;
        }
        
        .tab.active {
            background: rgba(255, 255, 255, 0.1);
            border-bottom-color: #4caf50;
        }
        
        .tab:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .tab-content {
            display: none;
            padding: 2rem;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .card h2 {
            margin-bottom: 1rem;
            font-size: 1.4rem;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 0.5rem;
        }
        
        .agent-list {
            max-height: 500px;
            overflow-y: auto;
        }
        
        .agent-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        
        .agent-item.paused {
            border-left-color: #ff9800;
            opacity: 0.8;
        }
        
        .agent-item.emergency-stopped {
            border-left-color: #f44336;
            opacity: 0.7;
        }
        
        .agent-info h3 {
            margin-bottom: 0.5rem;
        }
        
        .agent-info p {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .agent-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .control-btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .pause-btn {
            background: #ff9800;
            color: white;
        }
        
        .resume-btn {
            background: #4caf50;
            color: white;
        }
        
        .stop-btn {
            background: #f44336;
            color: white;
        }
        
        .override-btn {
            background: #9c27b0;
            color: white;
        }
        
        .approval-item {
            padding: 1rem;
            margin-bottom: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        
        .approval-item.high-risk {
            border-left-color: #f44336;
        }
        
        .approval-item.medium-risk {
            border-left-color: #ff9800;
        }
        
        .approval-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .approval-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .approve-btn {
            background: #4caf50;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        
        .reject-btn {
            background: #f44336;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        
        .alert-item {
            padding: 1rem;
            margin-bottom: 0.8rem;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }
        
        .alert-item.critical {
            background: rgba(244, 67, 54, 0.2);
        }
        
        .alert-item.high {
            background: rgba(255, 152, 0, 0.2);
        }
        
        .alert-item.medium {
            background: rgba(255, 193, 7, 0.1);
        }
        
        .alert-item.low {
            background: rgba(76, 175, 80, 0.1);
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-active {
            background: #4caf50;
        }
        
        .status-paused {
            background: #ff9800;
        }
        
        .status-stopped {
            background: #f44336;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        
        .modal-content {
            background-color: #fff;
            color: #333;
            margin: 15% auto;
            padding: 20px;
            border-radius: 15px;
            width: 80%;
            max-width: 500px;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #000;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        
        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ SutazAI Human Oversight Interface</h1>
        <div id="connectionStatus" class="connection-status">Connecting...</div>
        <div class="emergency-controls">
            <button class="emergency-btn bulk-pause" onclick="bulkPauseAgents()">⏸️ Pause All Agents</button>
            <button class="emergency-btn emergency-stop" onclick="systemEmergencyStop()">🛑 Emergency Stop</button>
        </div>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="showTab('overview')">📊 Overview</div>
        <div class="tab" onclick="showTab('agents')">🤖 Agent Control</div>
        <div class="tab" onclick="showTab('approvals')">✅ Approvals</div>
        <div class="tab" onclick="showTab('alerts')">⚠️ Alerts</div>
        <div class="tab" onclick="showTab('audit')">📋 Audit Trail</div>
    </div>
    
    <!-- Overview Tab -->
    <div id="overview" class="tab-content active">
        <div class="dashboard-grid">
            <div class="card">
                <h2>📈 System Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="totalAgents">-</div>
                        <div class="metric-label">Total Agents</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="activeAgents">-</div>
                        <div class="metric-label">Active</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="pausedAgents">-</div>
                        <div class="metric-label">Paused</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="stoppedAgents">-</div>
                        <div class="metric-label">Stopped</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🚨 Active Alerts</h2>
                <div id="overviewAlerts" class="alert-list">
                    <div class="loading">Loading alerts...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>⏳ Pending Approvals</h2>
                <div id="overviewApprovals" class="approval-list">
                    <div class="loading">Loading approvals...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>🔧 Active Overrides</h2>
                <div id="overviewOverrides" class="override-list">
                    <div class="loading">Loading overrides...</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Agent Control Tab -->
    <div id="agents" class="tab-content">
        <div class="card">
            <h2>🤖 Agent Control Panel</h2>
            <div class="agent-list" id="agentControlList">
                <div class="loading">Loading agents...</div>
            </div>
        </div>
    </div>
    
    <!-- Approvals Tab -->
    <div id="approvals" class="tab-content">
        <div class="card">
            <h2>✅ Approval Requests</h2>
            <div class="approval-list" id="approvalsList">
                <div class="loading">Loading approvals...</div>
            </div>
        </div>
    </div>
    
    <!-- Alerts Tab -->
    <div id="alerts" class="tab-content">
        <div class="card">
            <h2>⚠️ System Alerts</h2>
            <div class="alert-list" id="alertsList">
                <div class="loading">Loading alerts...</div>
            </div>
        </div>
    </div>
    
    <!-- Audit Trail Tab -->
    <div id="audit" class="tab-content">
        <div class="card">
            <h2>📋 Audit Trail</h2>
            <div class="audit-list" id="auditList">
                <div class="loading">Loading audit events...</div>
            </div>
        </div>
    </div>
    
    <!-- Override Modal -->
    <div id="overrideModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('overrideModal')">&times;</span>
            <h2>Create Override</h2>
            <form id="overrideForm">
                <div class="form-group">
                    <label for="overrideType">Override Type:</label>
                    <select id="overrideType" required>
                        <option value="parameter_change">Parameter Change</option>
                        <option value="decision_override">Decision Override</option>
                        <option value="policy_enforcement">Policy Enforcement</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="overrideDescription">Description:</label>
                    <textarea id="overrideDescription" required></textarea>
                </div>
                <div class="form-group">
                    <label for="overridePreviousValue">Previous Value:</label>
                    <input type="text" id="overridePreviousValue">
                </div>
                <div class="form-group">
                    <label for="overrideNewValue">New Value:</label>
                    <input type="text" id="overrideNewValue" required>
                </div>
                <div class="form-group">
                    <label for="overrideOperatorId">Operator ID:</label>
                    <input type="text" id="overrideOperatorId" required>
                </div>
                <button type="submit">Create Override</button>
            </form>
        </div>
    </div>
    
    <script>
        class HumanOversightInterface {
            constructor() {
                this.ws = null;
                this.isConnected = false;
                this.currentAgentId = null;
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.loadInitialData();
                
                // Setup form handlers
                document.getElementById('overrideForm').addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.submitOverride();
                });
                
                // Refresh data every 30 seconds
                setInterval(() => {
                    if (!this.isConnected) {
                        this.loadInitialData();
                    }
                }, 30000);
            }
            
            connectWebSocket() {
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
                
                try {
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        console.log('WebSocket connected');
                        this.isConnected = true;
                        this.updateConnectionStatus('connected');
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.updateInterface(data);
                        } catch (error) {
                            console.error('Error parsing WebSocket message:', error);
                        }
                    };
                    
                    this.ws.onclose = () => {
                        console.log('WebSocket disconnected');
                        this.isConnected = false;
                        this.updateConnectionStatus('disconnected');
                        setTimeout(() => this.connectWebSocket(), 5000);
                    };
                    
                } catch (error) {
                    console.error('Error creating WebSocket:', error);
                    setTimeout(() => this.connectWebSocket(), 5000);
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
                    default:
                        statusElement.textContent = '🟡 Connecting...';
                }
            }
            
            async loadInitialData() {
                try {
                    // Load all data endpoints
                    const [agents, overrides, approvals, alerts, audit] = await Promise.all([
                        fetch('/api/agents/status').then(r => r.json()),
                        fetch('/api/overrides').then(r => r.json()),
                        fetch('/api/approvals').then(r => r.json()),
                        fetch('/api/alerts').then(r => r.json()),
                        fetch('/api/audit').then(r => r.json())
                    ]);
                    
                    this.updateInterface({
                        agents: agents,
                        overrides: overrides,
                        approvals: approvals,
                        alerts: alerts,
                        audit: audit
                    });
                } catch (error) {
                    console.error('Error loading initial data:', error);
                }
            }
            
            updateInterface(data) {
                if (data.agents) this.updateAgentsList(data.agents);
                if (data.overrides) this.updateOverridesList(data.overrides);
                if (data.approvals) this.updateApprovalsList(data.approvals);
                if (data.alerts) this.updateAlertsList(data.alerts);
                if (data.audit) this.updateAuditList(data.audit);
            }
            
            updateAgentsList(agents) {
                const agentsList = document.getElementById('agentControlList');
                let html = '';
                
                for (const [agentId, agent] of Object.entries(agents)) {
                    const status = agent.status || 'active';
                    const statusClass = status === 'paused' ? 'paused' : 
                                       status === 'emergency_stopped' ? 'emergency-stopped' : '';
                    
                    html += `
                        <div class="agent-item ${statusClass}">
                            <div class="agent-info">
                                <h3>
                                    <span class="status-indicator status-${status}"></span>
                                    ${agentId}
                                </h3>
                                <p>Type: ${agent.type || 'unknown'} | Port: ${agent.port || 'N/A'}</p>
                                <p>Last Check: ${new Date(agent.last_check || Date.now()).toLocaleTimeString()}</p>
                            </div>
                            <div class="agent-controls">
                                ${status !== 'paused' ? 
                                    `<button class="control-btn pause-btn" onclick="oversight.pauseAgent('${agentId}')">⏸️ Pause</button>` :
                                    `<button class="control-btn resume-btn" onclick="oversight.resumeAgent('${agentId}')">▶️ Resume</button>`
                                }
                                <button class="control-btn stop-btn" onclick="oversight.emergencyStopAgent('${agentId}')">🛑 Stop</button>
                                <button class="control-btn override-btn" onclick="oversight.openOverrideModal('${agentId}')">⚙️ Override</button>
                            </div>
                        </div>
                    `;
                }
                
                agentsList.innerHTML = html || '<div class="loading">No agents found</div>';
                
                // Update metrics
                const totalAgents = Object.keys(agents).length;
                const activeAgents = Object.values(agents).filter(a => a.status === 'active').length;
                const pausedAgents = Object.values(agents).filter(a => a.status === 'paused').length;
                const stoppedAgents = Object.values(agents).filter(a => a.status === 'emergency_stopped').length;
                
                document.getElementById('totalAgents').textContent = totalAgents;
                document.getElementById('activeAgents').textContent = activeAgents;
                document.getElementById('pausedAgents').textContent = pausedAgents;
                document.getElementById('stoppedAgents').textContent = stoppedAgents;
            }
            
            updateApprovalsList(approvals) {
                const approvalsList = document.getElementById('approvalsList');
                const overviewApprovals = document.getElementById('overviewApprovals');
                let html = '';
                
                for (const approval of approvals.slice(0, 10)) {
                    const riskClass = approval.risk_level.toLowerCase() + '-risk';
                    const timeLeft = new Date(approval.expires_at) - new Date();
                    const timeLeftText = timeLeft > 0 ? `${Math.ceil(timeLeft / 60000)}m remaining` : 'EXPIRED';
                    
                    html += `
                        <div class="approval-item ${riskClass}">
                            <div class="approval-header">
                                <h3>${approval.request_type}</h3>
                                <div class="approval-controls">
                                    <button class="approve-btn" onclick="oversight.approveRequest('${approval.id}')">✅ Approve</button>
                                    <button class="reject-btn" onclick="oversight.rejectRequest('${approval.id}')">❌ Reject</button>
                                </div>
                            </div>
                            <p><strong>Agent:</strong> ${approval.agent_id}</p>
                            <p><strong>Description:</strong> ${approval.description}</p>
                            <p><strong>Risk Level:</strong> ${approval.risk_level.toUpperCase()}</p>
                            <p><strong>Time:</strong> ${timeLeftText}</p>
                        </div>
                    `;
                }
                
                const finalHtml = html || '<div style="text-align: center; opacity: 0.6;">🎉 No pending approvals</div>';
                approvalsList.innerHTML = finalHtml;
                overviewApprovals.innerHTML = finalHtml;
            }
            
            updateAlertsList(alerts) {
                const alertsList = document.getElementById('alertsList');
                const overviewAlerts = document.getElementById('overviewAlerts');
                let html = '';
                
                for (const alert of alerts.slice(0, 10)) {
                    const severityClass = alert.severity.toLowerCase();
                    const timestamp = new Date(alert.created_at).toLocaleTimeString();
                    
                    html += `
                        <div class="alert-item ${severityClass}">
                            <div class="alert-header">
                                <h3>${alert.alert_type}</h3>
                                <button class="control-btn" onclick="oversight.acknowledgeAlert('${alert.id}')">✅ Acknowledge</button>
                            </div>
                            <p><strong>Message:</strong> ${alert.message}</p>
                            <p><strong>Agent:</strong> ${alert.agent_id || 'System'}</p>
                            <p><strong>Severity:</strong> ${alert.severity.toUpperCase()}</p>
                            <p><strong>Time:</strong> ${timestamp}</p>
                        </div>
                    `;
                }
                
                const finalHtml = html || '<div style="text-align: center; opacity: 0.6;">🎉 No active alerts</div>';
                alertsList.innerHTML = finalHtml;
                overviewAlerts.innerHTML = finalHtml;
            }
            
            updateAuditList(events) {
                const auditList = document.getElementById('auditList');
                let html = '';
                
                for (const event of events.slice(0, 20)) {
                    const timestamp = new Date(event.timestamp).toLocaleString();
                    
                    html += `
                        <div class="audit-item">
                            <h3>${event.event_type}</h3>
                            <p><strong>Agent:</strong> ${event.agent_id}</p>
                            <p><strong>Operator:</strong> ${event.operator_id}</p>
                            <p><strong>Description:</strong> ${event.description}</p>
                            <p><strong>Time:</strong> ${timestamp}</p>
                        </div>
                    `;
                }
                
                auditList.innerHTML = html || '<div style="text-align: center; opacity: 0.6;">No audit events</div>';
            }
            
            async pauseAgent(agentId) {
                if (confirm(`Are you sure you want to pause agent ${agentId}?`)) {
                    try {
                        const response = await fetch(`/api/agents/${agentId}/pause`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                operator_id: prompt('Enter your operator ID:') || 'anonymous',
                                reason: prompt('Reason for pausing:') || 'Manual intervention'
                            })
                        });
                        
                        if (response.ok) {
                            this.loadInitialData();
                        } else {
                            alert('Failed to pause agent');
                        }
                    } catch (error) {
                        console.error('Error pausing agent:', error);
                        alert('Error pausing agent');
                    }
                }
            }
            
            async resumeAgent(agentId) {
                try {
                    const response = await fetch(`/api/agents/${agentId}/resume`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            operator_id: prompt('Enter your operator ID:') || 'anonymous'
                        })
                    });
                    
                    if (response.ok) {
                        this.loadInitialData();
                    } else {
                        alert('Failed to resume agent');
                    }
                } catch (error) {
                    console.error('Error resuming agent:', error);
                    alert('Error resuming agent');
                }
            }
            
            async emergencyStopAgent(agentId) {
                if (confirm(`⚠️ EMERGENCY STOP: Are you sure you want to emergency stop agent ${agentId}? This cannot be undone easily.`)) {
                    try {
                        const response = await fetch(`/api/agents/${agentId}/emergency_stop`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                operator_id: prompt('Enter your operator ID:') || 'anonymous',
                                reason: prompt('Emergency stop reason:') || 'Emergency intervention'
                            })
                        });
                        
                        if (response.ok) {
                            this.loadInitialData();
                        } else {
                            alert('Failed to emergency stop agent');
                        }
                    } catch (error) {
                        console.error('Error emergency stopping agent:', error);
                        alert('Error emergency stopping agent');
                    }
                }
            }
            
            openOverrideModal(agentId) {
                this.currentAgentId = agentId;
                document.getElementById('overrideModal').style.display = 'block';
            }
            
            async submitOverride() {
                const formData = {
                    override_type: document.getElementById('overrideType').value,
                    description: document.getElementById('overrideDescription').value,
                    previous_value: document.getElementById('overridePreviousValue').value,
                    new_value: document.getElementById('overrideNewValue').value,
                    operator_id: document.getElementById('overrideOperatorId').value
                };
                
                try {
                    const response = await fetch(`/api/agents/${this.currentAgentId}/override`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(formData)
                    });
                    
                    if (response.ok) {
                        document.getElementById('overrideModal').style.display = 'none';
                        document.getElementById('overrideForm').reset();
                        this.loadInitialData();
                    } else {
                        alert('Failed to create override');
                    }
                } catch (error) {
                    console.error('Error creating override:', error);
                    alert('Error creating override');
                }
            }
            
            async approveRequest(approvalId) {
                const note = prompt('Approval note (optional):');
                try {
                    const response = await fetch(`/api/approvals/${approvalId}/approve`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            approver_id: prompt('Enter your operator ID:') || 'anonymous',
                            note: note
                        })
                    });
                    
                    if (response.ok) {
                        this.loadInitialData();
                    } else {
                        alert('Failed to approve request');
                    }
                } catch (error) {
                    console.error('Error approving request:', error);
                    alert('Error approving request');
                }
            }
            
            async rejectRequest(approvalId) {
                const note = prompt('Rejection reason:');
                if (note) {
                    try {
                        const response = await fetch(`/api/approvals/${approvalId}/reject`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                approver_id: prompt('Enter your operator ID:') || 'anonymous',
                                note: note
                            })
                        });
                        
                        if (response.ok) {
                            this.loadInitialData();
                        } else {
                            alert('Failed to reject request');
                        }
                    } catch (error) {
                        console.error('Error rejecting request:', error);
                        alert('Error rejecting request');
                    }
                }
            }
            
            async acknowledgeAlert(alertId) {
                try {
                    const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            operator_id: prompt('Enter your operator ID:') || 'anonymous'
                        })
                    });
                    
                    if (response.ok) {
                        this.loadInitialData();
                    } else {
                        alert('Failed to acknowledge alert');
                    }
                } catch (error) {
                    console.error('Error acknowledging alert:', error);
                    alert('Error acknowledging alert');
                }
            }
        }
        
        function showTab(tabName) {
            // Hide all tab contents
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function closeModal(modalId) {
            document.getElementById(modalId).style.display = 'none';
        }
        
        async function bulkPauseAgents() {
            if (confirm('⚠️ Are you sure you want to pause ALL agents? This will affect the entire system.')) {
                try {
                    const response = await fetch('/api/agents/bulk/pause', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            operator_id: prompt('Enter your operator ID:') || 'anonymous',
                            reason: prompt('Reason for bulk pause:') || 'Bulk intervention'
                        })
                    });
                    
                    if (response.ok) {
                        oversight.loadInitialData();
                    } else {
                        alert('Failed to pause agents');
                    }
                } catch (error) {
                    console.error('Error pausing agents:', error);
                    alert('Error pausing agents');
                }
            }
        }
        
        async function systemEmergencyStop() {
            if (confirm('🚨 EMERGENCY STOP: This will immediately stop the entire SutazAI system. Are you absolutely sure?')) {
                if (confirm('This action cannot be undone easily. System recovery may take time. Proceed?')) {
                    try {
                        const response = await fetch('/api/system/emergency_stop', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                operator_id: prompt('Enter your operator ID:') || 'anonymous',
                                reason: prompt('Emergency stop reason:') || 'System emergency stop'
                            })
                        });
                        
                        if (response.ok) {
                            alert('Emergency stop initiated. System is shutting down.');
                            oversight.loadInitialData();
                        } else {
                            alert('Failed to initiate emergency stop');
                        }
                    } catch (error) {
                        console.error('Error initiating emergency stop:', error);
                        alert('Error initiating emergency stop');
                    }
                }
            }
        }
        
        // Initialize oversight interface
        const oversight = new HumanOversightInterface();
        
        // Close modals when clicking outside
        window.onclick = function(event) {
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
        """
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'human-oversight-interface'
        })
    
    async def get_agents_status(self, request):
        """Get current agent status"""
        try:
            # Load current agent status
            self._load_agent_data()
            
            # Enhance with control states
            enhanced_status = {}
            with sqlite3.connect(self.db_path) as conn:
                for agent_id, agent_info in self.agent_status.items():
                    cursor = conn.execute('''
                        SELECT is_paused, paused_by, paused_at, pause_reason,
                               emergency_stopped, stopped_by, stopped_at, stop_reason
                        FROM agent_control_states 
                        WHERE agent_id = ?
                    ''', (agent_id,))
                    
                    control_state = cursor.fetchone()
                    
                    enhanced_info = agent_info.copy()
                    if control_state:
                        enhanced_info.update({
                            'is_paused': bool(control_state[0]),
                            'paused_by': control_state[1],
                            'paused_at': control_state[2],
                            'pause_reason': control_state[3],
                            'emergency_stopped': bool(control_state[4]),
                            'stopped_by': control_state[5],
                            'stopped_at': control_state[6],
                            'stop_reason': control_state[7]
                        })
                        
                        # Update status based on control state
                        if enhanced_info['emergency_stopped']:
                            enhanced_info['status'] = 'emergency_stopped'
                        elif enhanced_info['is_paused']:
                            enhanced_info['status'] = 'paused'
                        else:
                            enhanced_info['status'] = 'active'
                    
                    enhanced_status[agent_id] = enhanced_info
            
            return web.json_response(enhanced_status)
            
        except Exception as e:
            logger.error(f"Error getting agents status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def pause_agent(self, request):
        """Pause a specific agent"""
        agent_id = request.match_info['agent_id']
        data = await request.json()
        
        try:
            # Create audit event
            audit_event = AuditEvent(
                id=str(uuid.uuid4()),
                event_type="agent_pause",
                agent_id=agent_id,
                operator_id=data.get('operator_id', 'anonymous'),
                description=f"Agent {agent_id} paused",
                before_state={'status': 'active'},
                after_state={'status': 'paused', 'reason': data.get('reason', '')},
                timestamp=datetime.utcnow(),
                compliance_tags=['agent_control', 'human_intervention']
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Update agent control state
                conn.execute('''
                    INSERT OR REPLACE INTO agent_control_states 
                    (agent_id, is_paused, paused_by, paused_at, pause_reason, last_update)
                    VALUES (?, 1, ?, ?, ?, ?)
                ''', (agent_id, data.get('operator_id'), datetime.utcnow().isoformat(), 
                      data.get('reason', ''), datetime.utcnow().isoformat()))
                
                # Store audit event
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                conn.commit()
            
            logger.info(f"Agent {agent_id} paused by {data.get('operator_id')}")
            
            return web.json_response({'status': 'success', 'message': f'Agent {agent_id} paused'})
            
        except Exception as e:
            logger.error(f"Error pausing agent {agent_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def resume_agent(self, request):
        """Resume a paused agent"""
        agent_id = request.match_info['agent_id']
        data = await request.json()
        
        try:
            # Create audit event
            audit_event = AuditEvent(
                id=str(uuid.uuid4()),
                event_type="agent_resume",
                agent_id=agent_id,
                operator_id=data.get('operator_id', 'anonymous'),
                description=f"Agent {agent_id} resumed",
                before_state={'status': 'paused'},
                after_state={'status': 'active'},
                timestamp=datetime.utcnow(),
                compliance_tags=['agent_control', 'human_intervention']
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Update agent control state
                conn.execute('''
                    INSERT OR REPLACE INTO agent_control_states 
                    (agent_id, is_paused, paused_by, paused_at, pause_reason, last_update)
                    VALUES (?, 0, NULL, NULL, NULL, ?)
                ''', (agent_id, datetime.utcnow().isoformat()))
                
                # Store audit event
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                conn.commit()
            
            logger.info(f"Agent {agent_id} resumed by {data.get('operator_id')}")
            
            return web.json_response({'status': 'success', 'message': f'Agent {agent_id} resumed'})
            
        except Exception as e:
            logger.error(f"Error resuming agent {agent_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def emergency_stop_agent(self, request):
        """Emergency stop a specific agent"""
        agent_id = request.match_info['agent_id']
        data = await request.json()
        
        try:
            # Create audit event
            audit_event = AuditEvent(
                id=str(uuid.uuid4()),
                event_type="agent_emergency_stop",
                agent_id=agent_id,
                operator_id=data.get('operator_id', 'anonymous'),
                description=f"Agent {agent_id} emergency stopped",
                before_state={'status': 'active'},
                after_state={'status': 'emergency_stopped', 'reason': data.get('reason', '')},
                timestamp=datetime.utcnow(),
                compliance_tags=['agent_control', 'emergency_intervention', 'critical']
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Update agent control state
                conn.execute('''
                    INSERT OR REPLACE INTO agent_control_states 
                    (agent_id, emergency_stopped, stopped_by, stopped_at, stop_reason, last_update)
                    VALUES (?, 1, ?, ?, ?, ?)
                ''', (agent_id, data.get('operator_id'), datetime.utcnow().isoformat(), 
                      data.get('reason', ''), datetime.utcnow().isoformat()))
                
                # Store audit event
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                # Create critical alert
                alert_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO alert_notifications 
                    (id, alert_type, severity, message, agent_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (alert_id, 'emergency_stop', 'critical', 
                      f'Agent {agent_id} emergency stopped by {data.get("operator_id")}',
                      agent_id, datetime.utcnow().isoformat()))
                
                conn.commit()
            
            logger.critical(f"Agent {agent_id} emergency stopped by {data.get('operator_id')}")
            
            return web.json_response({'status': 'success', 'message': f'Agent {agent_id} emergency stopped'})
            
        except Exception as e:
            logger.error(f"Error emergency stopping agent {agent_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def create_override(self, request):
        """Create a human override for an agent"""
        agent_id = request.match_info['agent_id']
        data = await request.json()
        
        try:
            override = HumanOverride(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                override_type=OverrideType(data['override_type']),
                description=data['description'],
                previous_value=data.get('previous_value'),
                new_value=data['new_value'],
                operator_id=data['operator_id'],
                timestamp=datetime.utcnow(),
                expiry_time=datetime.utcnow() + timedelta(hours=24)  # Default 24 hour expiry
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO human_overrides 
                    (id, agent_id, override_type, description, previous_value, new_value, 
                     operator_id, timestamp, expiry_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (override.id, override.agent_id, override.override_type.value,
                      override.description, json.dumps(override.previous_value),
                      json.dumps(override.new_value), override.operator_id,
                      override.timestamp.isoformat(), override.expiry_time.isoformat()))
                
                # Create audit event
                audit_event = AuditEvent(
                    id=str(uuid.uuid4()),
                    event_type="human_override_created",
                    agent_id=agent_id,
                    operator_id=data['operator_id'],
                    description=f"Override created: {data['description']}",
                    before_state={'value': override.previous_value},
                    after_state={'value': override.new_value},
                    timestamp=datetime.utcnow(),
                    compliance_tags=['human_override', 'parameter_change']
                )
                
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                conn.commit()
            
            self.active_overrides[override.id] = override
            
            logger.info(f"Override created for agent {agent_id} by {data['operator_id']}")
            
            return web.json_response({'status': 'success', 'override_id': override.id})
            
        except Exception as e:
            logger.error(f"Error creating override for agent {agent_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_active_overrides(self, request):
        """Get all active overrides"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, agent_id, override_type, description, previous_value, new_value,
                           operator_id, timestamp, expiry_time
                    FROM human_overrides 
                    WHERE is_active = 1 AND (expiry_time IS NULL OR expiry_time > ?)
                    ORDER BY timestamp DESC
                ''', (datetime.utcnow().isoformat(),))
                
                overrides = []
                for row in cursor.fetchall():
                    overrides.append({
                        'id': row[0],
                        'agent_id': row[1],
                        'override_type': row[2],
                        'description': row[3],
                        'previous_value': json.loads(row[4]) if row[4] else None,
                        'new_value': json.loads(row[5]) if row[5] else None,
                        'operator_id': row[6],
                        'timestamp': row[7],
                        'expiry_time': row[8]
                    })
                
                return web.json_response(overrides)
                
        except Exception as e:
            logger.error(f"Error getting active overrides: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_pending_approvals(self, request):
        """Get all pending approval requests"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, agent_id, request_type, description, context, risk_level,
                           created_at, expires_at, status
                    FROM approval_requests 
                    WHERE status = 'pending' AND expires_at > ?
                    ORDER BY risk_level DESC, created_at ASC
                ''', (datetime.utcnow().isoformat(),))
                
                approvals = []
                for row in cursor.fetchall():
                    approvals.append({
                        'id': row[0],
                        'agent_id': row[1],
                        'request_type': row[2],
                        'description': row[3],
                        'context': json.loads(row[4]),
                        'risk_level': row[5],
                        'created_at': row[6],
                        'expires_at': row[7],
                        'status': row[8]
                    })
                
                return web.json_response(approvals)
                
        except Exception as e:
            logger.error(f"Error getting pending approvals: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def approve_request(self, request):
        """Approve a pending request"""
        approval_id = request.match_info['approval_id']
        data = await request.json()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update approval status
                conn.execute('''
                    UPDATE approval_requests 
                    SET status = 'approved', approver_id = ?, approval_timestamp = ?, approval_note = ?
                    WHERE id = ?
                ''', (data.get('approver_id'), datetime.utcnow().isoformat(), 
                      data.get('note'), approval_id))
                
                # Create audit event
                audit_event = AuditEvent(
                    id=str(uuid.uuid4()),
                    event_type="approval_granted",
                    agent_id="system",
                    operator_id=data.get('approver_id', 'anonymous'),
                    description=f"Approval {approval_id} granted",
                    before_state={'status': 'pending'},
                    after_state={'status': 'approved', 'note': data.get('note')},
                    timestamp=datetime.utcnow(),
                    compliance_tags=['approval_workflow', 'human_decision']
                )
                
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                conn.commit()
            
            logger.info(f"Approval {approval_id} granted by {data.get('approver_id')}")
            
            return web.json_response({'status': 'success', 'message': 'Request approved'})
            
        except Exception as e:
            logger.error(f"Error approving request {approval_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def reject_request(self, request):
        """Reject a pending request"""
        approval_id = request.match_info['approval_id']
        data = await request.json()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update approval status
                conn.execute('''
                    UPDATE approval_requests 
                    SET status = 'rejected', approver_id = ?, approval_timestamp = ?, approval_note = ?
                    WHERE id = ?
                ''', (data.get('approver_id'), datetime.utcnow().isoformat(), 
                      data.get('note'), approval_id))
                
                # Create audit event
                audit_event = AuditEvent(
                    id=str(uuid.uuid4()),
                    event_type="approval_rejected",
                    agent_id="system",
                    operator_id=data.get('approver_id', 'anonymous'),
                    description=f"Approval {approval_id} rejected",
                    before_state={'status': 'pending'},
                    after_state={'status': 'rejected', 'note': data.get('note')},
                    timestamp=datetime.utcnow(),
                    compliance_tags=['approval_workflow', 'human_decision']
                )
                
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                conn.commit()
            
            logger.info(f"Approval {approval_id} rejected by {data.get('approver_id')}")
            
            return web.json_response({'status': 'success', 'message': 'Request rejected'})
            
        except Exception as e:
            logger.error(f"Error rejecting request {approval_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_active_alerts(self, request):
        """Get all active alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, alert_type, severity, message, agent_id, created_at,
                           acknowledged, acknowledged_by, acknowledged_at
                    FROM alert_notifications 
                    WHERE resolved = 0
                    ORDER BY severity DESC, created_at DESC
                    LIMIT 50
                ''')
                
                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'id': row[0],
                        'alert_type': row[1],
                        'severity': row[2],
                        'message': row[3],
                        'agent_id': row[4],
                        'created_at': row[5],
                        'acknowledged': bool(row[6]),
                        'acknowledged_by': row[7],
                        'acknowledged_at': row[8]
                    })
                
                return web.json_response(alerts)
                
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def acknowledge_alert(self, request):
        """Acknowledge an alert"""
        alert_id = request.match_info['alert_id']
        data = await request.json()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alert_notifications 
                    SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
                    WHERE id = ?
                ''', (data.get('operator_id'), datetime.utcnow().isoformat(), alert_id))
                
                conn.commit()
            
            logger.info(f"Alert {alert_id} acknowledged by {data.get('operator_id')}")
            
            return web.json_response({'status': 'success', 'message': 'Alert acknowledged'})
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_audit_events(self, request):
        """Get recent audit events"""
        try:
            limit = int(request.query.get('limit', 100))
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, event_type, agent_id, operator_id, description, 
                           before_state, after_state, timestamp, compliance_tags
                    FROM audit_events 
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                events = []
                for row in cursor.fetchall():
                    events.append({
                        'id': row[0],
                        'event_type': row[1],
                        'agent_id': row[2],
                        'operator_id': row[3],
                        'description': row[4],
                        'before_state': json.loads(row[5]) if row[5] else None,
                        'after_state': json.loads(row[6]) if row[6] else None,
                        'timestamp': row[7],
                        'compliance_tags': json.loads(row[8]) if row[8] else []
                    })
                
                return web.json_response(events)
                
        except Exception as e:
            logger.error(f"Error getting audit events: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def bulk_pause_agents(self, request):
        """Pause all agents"""
        data = await request.json()
        
        try:
            paused_agents = []
            
            with sqlite3.connect(self.db_path) as conn:
                for agent_id in self.agent_status.keys():
                    # Update agent control state
                    conn.execute('''
                        INSERT OR REPLACE INTO agent_control_states 
                        (agent_id, is_paused, paused_by, paused_at, pause_reason, last_update)
                        VALUES (?, 1, ?, ?, ?, ?)
                    ''', (agent_id, data.get('operator_id'), datetime.utcnow().isoformat(), 
                          data.get('reason', 'Bulk pause operation'), datetime.utcnow().isoformat()))
                    
                    paused_agents.append(agent_id)
                
                # Create audit event
                audit_event = AuditEvent(
                    id=str(uuid.uuid4()),
                    event_type="bulk_agent_pause",
                    agent_id="system",
                    operator_id=data.get('operator_id', 'anonymous'),
                    description=f"Bulk pause of {len(paused_agents)} agents",
                    before_state={'agents': list(self.agent_status.keys()), 'status': 'active'},
                    after_state={'agents': paused_agents, 'status': 'paused'},
                    timestamp=datetime.utcnow(),
                    compliance_tags=['bulk_operation', 'agent_control', 'critical']
                )
                
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                # Create critical alert
                alert_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO alert_notifications 
                    (id, alert_type, severity, message, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_id, 'bulk_pause', 'high', 
                      f'All {len(paused_agents)} agents paused by {data.get("operator_id")}',
                      datetime.utcnow().isoformat()))
                
                conn.commit()
            
            logger.warning(f"Bulk pause of {len(paused_agents)} agents by {data.get('operator_id')}")
            
            return web.json_response({
                'status': 'success', 
                'message': f'{len(paused_agents)} agents paused',
                'paused_agents': paused_agents
            })
            
        except Exception as e:
            logger.error(f"Error in bulk pause operation: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def system_emergency_stop(self, request):
        """Emergency stop the entire system"""
        data = await request.json()
        
        try:
            stopped_agents = []
            
            with sqlite3.connect(self.db_path) as conn:
                for agent_id in self.agent_status.keys():
                    # Update agent control state
                    conn.execute('''
                        INSERT OR REPLACE INTO agent_control_states 
                        (agent_id, emergency_stopped, stopped_by, stopped_at, stop_reason, last_update)
                        VALUES (?, 1, ?, ?, ?, ?)
                    ''', (agent_id, data.get('operator_id'), datetime.utcnow().isoformat(), 
                          data.get('reason', 'System emergency stop'), datetime.utcnow().isoformat()))
                    
                    stopped_agents.append(agent_id)
                
                # Create audit event
                audit_event = AuditEvent(
                    id=str(uuid.uuid4()),
                    event_type="system_emergency_stop",
                    agent_id="system",
                    operator_id=data.get('operator_id', 'anonymous'),
                    description=f"System emergency stop - {len(stopped_agents)} agents stopped",
                    before_state={'system_status': 'operational', 'agents': list(self.agent_status.keys())},
                    after_state={'system_status': 'emergency_stopped', 'agents': stopped_agents},
                    timestamp=datetime.utcnow(),
                    compliance_tags=['emergency_stop', 'system_control', 'critical']
                )
                
                conn.execute('''
                    INSERT INTO audit_events 
                    (id, event_type, agent_id, operator_id, description, before_state, after_state, timestamp, compliance_tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (audit_event.id, audit_event.event_type, audit_event.agent_id,
                      audit_event.operator_id, audit_event.description, 
                      json.dumps(audit_event.before_state), json.dumps(audit_event.after_state),
                      audit_event.timestamp.isoformat(), json.dumps(audit_event.compliance_tags)))
                
                # Create critical alert
                alert_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO alert_notifications 
                    (id, alert_type, severity, message, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_id, 'system_emergency_stop', 'critical', 
                      f'SYSTEM EMERGENCY STOP initiated by {data.get("operator_id")} - All agents stopped',
                      datetime.utcnow().isoformat()))
                
                conn.commit()
            
            logger.critical(f"SYSTEM EMERGENCY STOP initiated by {data.get('operator_id')} - {len(stopped_agents)} agents stopped")
            
            
            return web.json_response({
                'status': 'success', 
                'message': f'System emergency stop initiated - {len(stopped_agents)} agents stopped',
                'stopped_agents': stopped_agents
            })
            
        except Exception as e:
            logger.error(f"Error in system emergency stop: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        logger.info("New WebSocket connection established for oversight interface")
        
        try:
            # Send initial data
            initial_data = {
                'agents': await self.get_agents_status(request),
                'overrides': await self.get_active_overrides(request),
                'approvals': await self.get_pending_approvals(request),
                'alerts': await self.get_active_alerts(request),
                'audit': await self.get_audit_events(request)
            }
            
            await ws.send_str(json.dumps(initial_data))
            
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
                    # Gather all data
                    update_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'agents': await self.get_agents_status(None),
                        'overrides': await self.get_active_overrides(None),
                        'approvals': await self.get_pending_approvals(None),
                        'alerts': await self.get_active_alerts(None)
                    }
                    
                    message = json.dumps(update_data)
                    
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
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(30)
    
    async def start(self):
        """Start the oversight interface server"""
        logger.info(f"Starting Human Oversight Interface on port {self.port}")
        
        # Start WebSocket broadcasting
        broadcast_task = asyncio.create_task(self.broadcast_updates())
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Human Oversight Interface available at http://localhost:{self.port}")
        
        try:
            await broadcast_task
        except asyncio.CancelledError:
            logger.info("Oversight interface tasks cancelled")
        finally:
            await runner.cleanup()
    
    def stop(self):
        """Stop the oversight interface"""
        self.shutdown_event.set()


async def main():
    """Main entry point"""
    interface = HumanOversightInterface()
    
    try:
        await interface.start()
    except KeyboardInterrupt:
        logger.info("Human Oversight Interface stopped by user")
    except Exception as e:
        logger.error(f"Human Oversight Interface error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())