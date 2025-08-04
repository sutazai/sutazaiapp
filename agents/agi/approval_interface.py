#!/usr/bin/env python3
"""
Owner Approval Interface for AGI/ASI Self-Improvement
Provides web interface and API for reviewing and approving system improvements

Features:
- Web dashboard for reviewing proposals
- Detailed analysis and risk assessment
- One-click approval/rejection
- Audit trail and rollback capabilities
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collective_intelligence import CollectiveIntelligence, ImprovementStatus

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ApprovalDecision(BaseModel):
    """Owner's decision on an improvement proposal"""
    proposal_id: str
    approved: bool
    feedback: Optional[str] = None
    owner_notes: Optional[str] = None


class ApprovalInterface:
    """Web interface for AGI/ASI owner approval system"""
    
    def __init__(self, 
                 collective_intelligence: CollectiveIntelligence,
                 port: int = 8888,
                 host: str = "0.0.0.0"):
        
        self.collective = collective_intelligence
        self.port = port
        self.host = host
        
        # Setup FastAPI app
        self.app = FastAPI(title="SutazAI AGI/ASI Approval System")
        self._setup_routes()
        self._setup_middleware()
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Approval history
        self.approval_history: List[Dict[str, Any]] = []
        self._load_approval_history()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard"""
            return self._generate_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_collective_status():
            """Get current collective intelligence status"""
            return await self.collective.get_collective_status()
        
        @self.app.get("/api/proposals/pending")
        async def get_pending_proposals():
            """Get all proposals awaiting approval"""
            pending = [
                self._serialize_proposal(pid, proposal)
                for pid, proposal in self.collective.improvement_proposals.items()
                if proposal.status == ImprovementStatus.TESTING
            ]
            return sorted(pending, key=lambda x: x["created_at"], reverse=True)
        
        @self.app.get("/api/proposals/all")
        async def get_all_proposals():
            """Get all improvement proposals"""
            all_proposals = [
                self._serialize_proposal(pid, proposal)
                for pid, proposal in self.collective.improvement_proposals.items()
            ]
            return sorted(all_proposals, key=lambda x: x["created_at"], reverse=True)
        
        @self.app.get("/api/proposals/{proposal_id}")
        async def get_proposal_details(proposal_id: str):
            """Get detailed information about a specific proposal"""
            if proposal_id not in self.collective.improvement_proposals:
                raise HTTPException(status_code=404, detail="Proposal not found")
            
            proposal = self.collective.improvement_proposals[proposal_id]
            return self._serialize_proposal(proposal_id, proposal, detailed=True)
        
        @self.app.post("/api/proposals/{proposal_id}/approve")
        async def approve_proposal(proposal_id: str, decision: ApprovalDecision):
            """Approve or reject a proposal"""
            if proposal_id not in self.collective.improvement_proposals:
                raise HTTPException(status_code=404, detail="Proposal not found")
            
            # Process decision
            await self.collective.process_owner_decision(
                proposal_id=proposal_id,
                approved=decision.approved,
                feedback=decision.feedback
            )
            
            # Record in history
            self._record_approval_decision(proposal_id, decision)
            
            # Notify WebSocket clients
            await self._notify_websocket_clients({
                "type": "proposal_decision",
                "proposal_id": proposal_id,
                "approved": decision.approved
            })
            
            return {"status": "success", "proposal_id": proposal_id}
        
        @self.app.get("/api/history")
        async def get_approval_history():
            """Get approval history"""
            return self.approval_history[-100:]  # Last 100 decisions
        
        @self.app.get("/api/metrics")
        async def get_approval_metrics():
            """Get approval metrics and statistics"""
            total_proposals = len(self.collective.improvement_proposals)
            approved = len([p for p in self.collective.improvement_proposals.values() 
                          if p.status == ImprovementStatus.APPROVED])
            rejected = len([p for p in self.collective.improvement_proposals.values() 
                          if p.status == ImprovementStatus.REJECTED])
            pending = len([p for p in self.collective.improvement_proposals.values() 
                         if p.status == ImprovementStatus.TESTING])
            
            # Calculate approval rate
            decided = approved + rejected
            approval_rate = (approved / decided * 100) if decided > 0 else 0
            
            # Performance impact of approved improvements
            performance_gains = []
            for proposal in self.collective.improvement_proposals.values():
                if proposal.status == ImprovementStatus.APPLIED and proposal.test_results:
                    gain = proposal.test_results.get("performance_tests", {}).get("gain_percent", 0)
                    performance_gains.append(gain)
            
            avg_performance_gain = sum(performance_gains) / len(performance_gains) if performance_gains else 0
            
            return {
                "total_proposals": total_proposals,
                "approved": approved,
                "rejected": rejected,
                "pending": pending,
                "approval_rate": approval_rate,
                "avg_performance_gain": avg_performance_gain,
                "total_rollbacks": len(self.collective.rollback_history)
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Send initial status
                status = await self.collective.get_collective_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status
                })
                
                # Keep connection alive
                while True:
                    data = await websocket.receive_text()
                    # Echo back for now
                    await websocket.send_text(f"Echo: {data}")
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.remove(websocket)
        
        @self.app.post("/api/emergency-stop")
        async def emergency_stop():
            """Trigger emergency stop of the collective"""
            self.collective.emergency_stop.set()
            
            # Notify all connections
            await self._notify_websocket_clients({
                "type": "emergency_stop",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {"status": "emergency_stop_triggered"}
    
    def _serialize_proposal(self, proposal_id: str, proposal, detailed: bool = False) -> Dict[str, Any]:
        """Serialize proposal for API response"""
        data = {
            "proposal_id": proposal_id,
            "agent_name": proposal.agent_name,
            "improvement_type": proposal.improvement_type,
            "description": proposal.description,
            "rationale": proposal.rationale,
            "expected_benefit": proposal.expected_benefit,
            "risk_assessment": proposal.risk_assessment,
            "status": proposal.status.value,
            "confidence_score": proposal.confidence_score,
            "created_at": proposal.created_at.isoformat(),
            "consensus_score": self.collective._calculate_consensus_score(proposal.consensus_votes)
        }
        
        if detailed:
            data.update({
                "test_results": proposal.test_results,
                "consensus_votes": proposal.consensus_votes,
                "code_changes": proposal.code_changes,
                "reviewed_at": proposal.reviewed_at.isoformat() if proposal.reviewed_at else None,
                "applied_at": proposal.applied_at.isoformat() if proposal.applied_at else None
            })
        
        return data
    
    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI AGI/ASI Approval System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px 0;
            margin-bottom: 30px;
            border-bottom: 2px solid #0f3460;
        }
        
        h1 {
            text-align: center;
            color: #4fbdba;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #7ec8e3;
            font-size: 1.1em;
        }
        
        .status-bar {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .status-item {
            text-align: center;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
            border: 1px solid #0f3460;
        }
        
        .status-value {
            font-size: 2em;
            font-weight: bold;
            color: #4fbdba;
        }
        
        .status-label {
            color: #7ec8e3;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .proposals-section {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .section-title {
            color: #4fbdba;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 2px solid #0f3460;
            padding-bottom: 10px;
        }
        
        .proposal-card {
            background: #16213e;
            border: 1px solid #0f3460;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .proposal-card:hover {
            border-color: #4fbdba;
            box-shadow: 0 4px 20px rgba(79, 189, 186, 0.2);
        }
        
        .proposal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .proposal-id {
            color: #7ec8e3;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        .proposal-type {
            background: #0f3460;
            color: #4fbdba;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        
        .proposal-description {
            color: #e0e0e0;
            margin-bottom: 15px;
        }
        
        .proposal-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric {
            background: #0a0a0a;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #4fbdba;
        }
        
        .metric-label {
            font-size: 0.8em;
            color: #7ec8e3;
        }
        
        .proposal-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .approve-btn {
            background: #27ae60;
            color: white;
        }
        
        .approve-btn:hover {
            background: #2ecc71;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        }
        
        .reject-btn {
            background: #c0392b;
            color: white;
        }
        
        .reject-btn:hover {
            background: #e74c3c;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }
        
        .details-btn {
            background: #2980b9;
            color: white;
        }
        
        .details-btn:hover {
            background: #3498db;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        
        .emergency-stop {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #e74c3c;
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: bold;
            box-shadow: 0 4px 20px rgba(231, 76, 60, 0.4);
            z-index: 1000;
        }
        
        .emergency-stop:hover {
            background: #c0392b;
            transform: scale(1.05);
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 2000;
        }
        
        .modal-content {
            position: relative;
            background: #1a1a2e;
            margin: 50px auto;
            padding: 30px;
            width: 90%;
            max-width: 800px;
            border-radius: 10px;
            border: 2px solid #0f3460;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .close-modal {
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 2em;
            color: #7ec8e3;
            cursor: pointer;
        }
        
        .test-results {
            background: #0a0a0a;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-active { background: #27ae60; }
        .status-pending { background: #f39c12; }
        .status-error { background: #e74c3c; }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>🧠 SutazAI AGI/ASI Control Center</h1>
            <p class="subtitle">Collective Intelligence Oversight & Approval System</p>
        </div>
    </header>
    
    <div class="container">
        <div class="status-bar" id="statusBar">
            <div class="status-item">
                <div class="status-value" id="collectiveAwareness">0.00</div>
                <div class="status-label">Collective Awareness</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="activeAgents">0</div>
                <div class="status-label">Active Agents</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="pendingApprovals">0</div>
                <div class="status-label">Pending Approvals</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="successRate">0%</div>
                <div class="status-label">Success Rate</div>
            </div>
        </div>
        
        <div class="proposals-section">
            <h2 class="section-title">
                <span class="status-indicator status-pending pulse"></span>
                Pending Improvement Proposals
            </h2>
            <div id="pendingProposals">
                <p style="text-align: center; color: #7ec8e3;">Loading proposals...</p>
            </div>
        </div>
        
        <div class="proposals-section">
            <h2 class="section-title">
                <span class="status-indicator status-active"></span>
                Recent Decisions
            </h2>
            <div id="recentDecisions">
                <p style="text-align: center; color: #7ec8e3;">Loading history...</p>
            </div>
        </div>
    </div>
    
    <button class="emergency-stop" onclick="emergencyStop()">
        🛑 EMERGENCY STOP
    </button>
    
    <div id="detailsModal" class="modal">
        <div class="modal-content">
            <span class="close-modal" onclick="closeModal()">&times;</span>
            <div id="modalContent"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        
        // Initialize WebSocket connection
        function initWebSocket() {
            const wsUrl = `ws://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(initWebSocket, 5000);
            };
        }
        
        // Handle WebSocket messages
        function handleWebSocketMessage(data) {
            if (data.type === 'status_update') {
                updateStatus(data.data);
            } else if (data.type === 'proposal_decision') {
                loadPendingProposals();
                loadRecentDecisions();
            }
        }
        
        // Update status bar
        async function updateStatus(statusData) {
            if (!statusData) {
                const response = await fetch('/api/status');
                statusData = await response.json();
            }
            
            document.getElementById('collectiveAwareness').textContent = 
                statusData.collective_awareness.toFixed(2);
            document.getElementById('activeAgents').textContent = 
                statusData.active_agents;
            document.getElementById('pendingApprovals').textContent = 
                statusData.pending_approvals;
            
            const metrics = statusData.performance_metrics;
            if (metrics) {
                document.getElementById('successRate').textContent = 
                    (metrics.success_rate * 100).toFixed(1) + '%';
            }
        }
        
        // Load pending proposals
        async function loadPendingProposals() {
            const response = await fetch('/api/proposals/pending');
            const proposals = await response.json();
            
            const container = document.getElementById('pendingProposals');
            
            if (proposals.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #7ec8e3;">No pending proposals</p>';
                return;
            }
            
            container.innerHTML = proposals.map(proposal => `
                <div class="proposal-card">
                    <div class="proposal-header">
                        <span class="proposal-id">${proposal.proposal_id}</span>
                        <span class="proposal-type">${proposal.improvement_type}</span>
                    </div>
                    <div class="proposal-description">
                        <strong>${proposal.description}</strong><br>
                        ${proposal.rationale}
                    </div>
                    <div class="proposal-metrics">
                        <div class="metric">
                            <div class="metric-value">${(proposal.expected_benefit * 100).toFixed(0)}%</div>
                            <div class="metric-label">Expected Benefit</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${(proposal.risk_assessment * 100).toFixed(0)}%</div>
                            <div class="metric-label">Risk Level</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${(proposal.confidence_score * 100).toFixed(0)}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${(proposal.consensus_score * 100).toFixed(0)}%</div>
                            <div class="metric-label">Consensus</div>
                        </div>
                    </div>
                    <div class="proposal-actions">
                        <button class="details-btn" onclick="showDetails('${proposal.proposal_id}')">
                            View Details
                        </button>
                        <button class="approve-btn" onclick="approveProposal('${proposal.proposal_id}')">
                            ✓ Approve
                        </button>
                        <button class="reject-btn" onclick="rejectProposal('${proposal.proposal_id}')">
                            ✗ Reject
                        </button>
                    </div>
                </div>
            `).join('');
        }
        
        // Load recent decisions
        async function loadRecentDecisions() {
            const response = await fetch('/api/history');
            const history = await response.json();
            
            const container = document.getElementById('recentDecisions');
            
            if (history.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #7ec8e3;">No recent decisions</p>';
                return;
            }
            
            container.innerHTML = history.slice(-5).reverse().map(decision => `
                <div class="proposal-card" style="opacity: 0.8;">
                    <div class="proposal-header">
                        <span class="proposal-id">${decision.proposal_id}</span>
                        <span class="proposal-type" style="background: ${decision.approved ? '#27ae60' : '#c0392b'}">
                            ${decision.approved ? 'APPROVED' : 'REJECTED'}
                        </span>
                    </div>
                    <div class="proposal-description">
                        <strong>Decision Time:</strong> ${new Date(decision.timestamp).toLocaleString()}<br>
                        ${decision.feedback ? `<strong>Feedback:</strong> ${decision.feedback}` : ''}
                    </div>
                </div>
            `).join('');
        }
        
        // Show proposal details
        async function showDetails(proposalId) {
            const response = await fetch(`/api/proposals/${proposalId}`);
            const proposal = await response.json();
            
            const modal = document.getElementById('detailsModal');
            const content = document.getElementById('modalContent');
            
            content.innerHTML = `
                <h2>Proposal Details: ${proposal.proposal_id}</h2>
                <div style="margin-top: 20px;">
                    <h3>Description</h3>
                    <p>${proposal.description}</p>
                    
                    <h3>Rationale</h3>
                    <p>${proposal.rationale}</p>
                    
                    <h3>Test Results</h3>
                    <div class="test-results">
${JSON.stringify(proposal.test_results, null, 2)}
                    </div>
                    
                    <h3>Code Changes</h3>
                    <div class="test-results">
${JSON.stringify(proposal.code_changes, null, 2)}
                    </div>
                    
                    <h3>Agent Consensus</h3>
                    <div class="test-results">
${JSON.stringify(proposal.consensus_votes, null, 2)}
                    </div>
                </div>
            `;
            
            modal.style.display = 'block';
        }
        
        // Close modal
        function closeModal() {
            document.getElementById('detailsModal').style.display = 'none';
        }
        
        // Approve proposal
        async function approveProposal(proposalId) {
            const feedback = prompt('Any feedback for this approval? (optional)');
            
            const response = await fetch(`/api/proposals/${proposalId}/approve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    proposal_id: proposalId,
                    approved: true,
                    feedback: feedback
                })
            });
            
            if (response.ok) {
                alert('Proposal approved successfully!');
                loadPendingProposals();
                loadRecentDecisions();
                updateStatus();
            }
        }
        
        // Reject proposal
        async function rejectProposal(proposalId) {
            const feedback = prompt('Please provide a reason for rejection:');
            
            if (!feedback) {
                alert('Rejection reason is required');
                return;
            }
            
            const response = await fetch(`/api/proposals/${proposalId}/approve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    proposal_id: proposalId,
                    approved: false,
                    feedback: feedback
                })
            });
            
            if (response.ok) {
                alert('Proposal rejected');
                loadPendingProposals();
                loadRecentDecisions();
                updateStatus();
            }
        }
        
        // Emergency stop
        async function emergencyStop() {
            if (confirm('Are you sure you want to trigger an EMERGENCY STOP? This will halt all AGI/ASI operations.')) {
                const response = await fetch('/api/emergency-stop', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    alert('EMERGENCY STOP TRIGGERED - All operations halted');
                    document.body.style.background = '#8b0000';
                }
            }
        }
        
        // Auto-refresh
        setInterval(() => {
            updateStatus();
            loadPendingProposals();
        }, 30000); // Every 30 seconds
        
        // Initialize on load
        window.onload = () => {
            initWebSocket();
            updateStatus();
            loadPendingProposals();
            loadRecentDecisions();
        };
        
        // Close modal on click outside
        window.onclick = (event) => {
            const modal = document.getElementById('detailsModal');
            if (event.target === modal) {
                closeModal();
            }
        };
    </script>
</body>
</html>
        """
    
    def _record_approval_decision(self, proposal_id: str, decision: ApprovalDecision):
        """Record approval decision in history"""
        record = {
            "proposal_id": proposal_id,
            "approved": decision.approved,
            "feedback": decision.feedback,
            "owner_notes": decision.owner_notes,
            "timestamp": datetime.utcnow().isoformat(),
            "proposal_summary": {
                "type": self.collective.improvement_proposals[proposal_id].improvement_type,
                "expected_benefit": self.collective.improvement_proposals[proposal_id].expected_benefit,
                "risk_assessment": self.collective.improvement_proposals[proposal_id].risk_assessment
            }
        }
        
        self.approval_history.append(record)
        self._save_approval_history()
    
    async def _notify_websocket_clients(self, message: Dict[str, Any]):
        """Notify all connected WebSocket clients"""
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to websocket: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
    
    def _save_approval_history(self):
        """Save approval history to disk"""
        try:
            history_file = self.collective.data_path / "approval_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.approval_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save approval history: {e}")
    
    def _load_approval_history(self):
        """Load approval history from disk"""
        try:
            history_file = self.collective.data_path / "approval_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.approval_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load approval history: {e}")
    
    async def run(self):
        """Run the approval interface server"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main entry point"""
    # Initialize collective intelligence
    collective = CollectiveIntelligence()
    await collective.awaken()
    
    # Create and run approval interface
    interface = ApprovalInterface(collective)
    
    # Run both in parallel
    await asyncio.gather(
        collective.run_async(),
        interface.run()
    )


if __name__ == "__main__":
    asyncio.run(main())