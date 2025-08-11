"""
Unified Orchestration API for SutazAI automation/advanced automation System
===================================================

A comprehensive FastAPI-based REST API that provides unified access to the entire
orchestration system, including agent management, task coordination, infrastructure
control, monitoring, and system management for all 38 AI agents.

Key Features:
- RESTful API for all orchestration functions
- Real-time WebSocket connections for monitoring
- Agent lifecycle management endpoints
- Task submission and tracking
- System health and metrics
- Infrastructure control and scaling
- Configuration management
- Authentication and authorization
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis.asyncio as redis

from .master_agent_orchestrator import MasterAgentOrchestrator, Task, AgentCapability, TaskPriority, CoordinationPattern
from .advanced_message_bus import AdvancedMessageBus, Message, MessageType, CommunicationPattern, MessagePriority
from .infrastructure_integration import InfrastructureIntegration
from .orchestration_dashboard import OrchestrationDashboard

logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

class TaskRequest(BaseModel):
    """Task submission request model"""
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    type: str = Field(default="general", description="Task type")
    priority: str = Field(default="medium", description="Task priority")
    requirements: List[str] = Field(default_factory=list, description="Required capabilities")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Task constraints")
    coordination_pattern: Optional[str] = Field(None, description="Preferred coordination pattern")
    deadline: Optional[datetime] = Field(None, description="Task deadline")


class TaskResponse(BaseModel):
    """Task response model"""
    session_id: str
    task_id: str
    status: str
    message: str


class AgentStatusResponse(BaseModel):
    """Agent status response model"""
    agent_id: str
    name: str
    type: str
    status: str
    capabilities: List[str]
    current_load: int
    success_rate: float
    response_time: float
    last_health_check: Optional[datetime]


class SystemStatusResponse(BaseModel):
    """System status response model"""
    orchestrator_status: str
    total_agents: int
    healthy_agents: int
    active_tasks: int
    active_sessions: int
    system_health: float
    message_throughput: int
    infrastructure_status: Dict[str, Any]


class MessageRequest(BaseModel):
    """Message request model"""
    recipient_id: Optional[str] = None
    message_type: str = "chat_message"
    pattern: str = "point_to_point"
    priority: str = "normal"
    topic: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    expires_in_seconds: Optional[int] = None


class ServiceControlRequest(BaseModel):
    """Service control request model"""
    action: str  # start, stop, restart, scale
    instances: Optional[int] = None


class ConfigurationRequest(BaseModel):
    """Configuration update request model"""
    section: str
    key: str
    value: Any


# ==================== WebSocket Connection Manager ====================

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = client_info or {}
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")
                disconnected_connections.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected_connections:
            self.disconnect(connection)


# ==================== Unified Orchestration API ====================

class UnifiedOrchestrationAPI:
    """
    Comprehensive orchestration API combining all system components
    """
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_url = redis_url
        
        # Initialize core components
        self.orchestrator = MasterAgentOrchestrator(redis_url)
        self.message_bus = AdvancedMessageBus(redis_url)
        self.infrastructure = InfrastructureIntegration(redis_url)
        self.dashboard = OrchestrationDashboard(self.orchestrator, self.message_bus)
        
        # WebSocket manager
        self.connection_manager = ConnectionManager()
        
        # FastAPI app
        self.app = FastAPI(
            title="SutazAI Orchestration API",
            description="Unified API for managing 38 AI agents and orchestration system",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Security
        self.security = HTTPBearer()
        
        # Configuration
        self.config = {
            "enable_auth": False,  # Set to True for production
            "max_concurrent_tasks": 100,
            "websocket_heartbeat_interval": 30,
            "api_rate_limit": 1000  # requests per minute
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("Unified Orchestration API initialized")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
        "http://localhost:10011",  # Frontend Streamlit UI
        "http://localhost:10010",  # Backend API
        "http://127.0.0.1:10011",  # Alternative localhost
        "http://127.0.0.1:10010",  # Alternative localhost
    ],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["Accept", "Accept-Language", "Content-Type", "Content-Language", "Authorization", "X-Requested-With", "X-CSRFToken", "Cache-Control"],
        )
    
    def _setup_routes(self):
        """Setup all API routes - main orchestrator."""
        self._setup_health_routes()
        self._setup_agent_management_routes()
        self._setup_task_management_routes()
        self._setup_message_bus_routes()
        self._setup_infrastructure_routes()
        self._setup_configuration_routes()
        self._setup_monitoring_routes()
        self._setup_websocket_routes()
        
    def _setup_health_routes(self):
        """Setup health and status endpoints."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0"
            }
        
        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get comprehensive system status"""
            return await self._get_comprehensive_system_status()
            
    async def _get_comprehensive_system_status(self) -> SystemStatusResponse:
        """Helper to get comprehensive system status."""
        try:
            orchestrator_status = await self.orchestrator.get_system_status()
            infrastructure_status = await self.infrastructure.get_system_status()
            message_bus_status = await self.message_bus.get_system_status()
            
            return SystemStatusResponse(
                orchestrator_status=orchestrator_status["orchestrator_status"],
                total_agents=orchestrator_status["total_agents"],
                healthy_agents=orchestrator_status["healthy_agents"],
                active_tasks=orchestrator_status["active_tasks"],
                active_sessions=orchestrator_status["active_sessions"],
                system_health=orchestrator_status["healthy_agents"] / max(orchestrator_status["total_agents"], 1),
                message_throughput=message_bus_status.get("performance_metrics", {}).get("total_messages_sent", 0),
                infrastructure_status=infrastructure_status
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")
            
    def _setup_agent_management_routes(self):
        """Setup agent management endpoints."""
        
        # ==================== Agent Management Endpoints ====================
        
        @self.app.get("/agents")
        async def list_agents(
            status: Optional[str] = None,
            capability: Optional[str] = None,
            agent_type: Optional[str] = None
        ):
            """List all agents with optional filtering"""
            try:
                agent_status = await self.orchestrator.get_agent_status()
                
                agents = []
                for agent_id, agent_info in agent_status.items():
                    # Apply filters
                    if status and agent_info.get("status") != status:
                        continue
                    if capability and capability not in agent_info.get("capabilities", []):
                        continue
                    if agent_type and agent_info.get("type") != agent_type:
                        continue
                    
                    agents.append({
                        "agent_id": agent_id,
                        **agent_info
                    })
                
                return {"agents": agents, "count": len(agents)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")
        
        @self.app.get("/agents/{agent_id}", response_model=AgentStatusResponse)
        async def get_agent_status(agent_id: str):
            """Get detailed status for a specific agent"""
            try:
                agent_info = await self.orchestrator.get_agent_status(agent_id)
                if not agent_info:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                
                return AgentStatusResponse(
                    agent_id=agent_id,
                    name=agent_info["name"],
                    type=agent_info["type"],
                    status=agent_info["status"],
                    capabilities=agent_info["capabilities"],
                    current_load=agent_info["current_load"],
                    success_rate=agent_info.get("success_rate", 1.0),
                    response_time=agent_info.get("response_time", 1.0),
                    last_health_check=agent_info.get("last_health_check")
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")
        
        @self.app.post("/agents/{agent_id}/control")
        async def control_agent(agent_id: str, request: ServiceControlRequest):
            """Control agent lifecycle (start, stop, restart, scale)"""
            try:
                if request.action == "start":
                    success = await self.infrastructure.deploy_service(agent_id)
                elif request.action == "stop":
                    success = await self.infrastructure.stop_service(agent_id)
                elif request.action == "restart":
                    success = await self.infrastructure.restart_service(agent_id)
                elif request.action == "scale":
                    if request.instances is None:
                        raise HTTPException(status_code=400, detail="instances parameter required for scale action")
                    success = await self.infrastructure.scale_service(agent_id, request.instances)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
                
                if success:
                    return {"message": f"Successfully {request.action}ed agent {agent_id}"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to {request.action} agent {agent_id}")
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Agent control failed: {str(e)}")
        
        # ==================== Task Management Endpoints ====================
        
        @self.app.post("/tasks", response_model=TaskResponse)
        async def submit_task(task_request: TaskRequest):
            """Submit a new task for orchestration"""
            try:
                # Convert request to task data
                task_data = {
                    "name": task_request.name,
                    "description": task_request.description,
                    "type": task_request.type,
                    "priority": task_request.priority,
                    "requirements": task_request.requirements,
                    "payload": task_request.payload,
                    "constraints": task_request.constraints
                }
                
                if task_request.deadline:
                    task_data["deadline"] = task_request.deadline
                
                # Submit task
                session_id = await self.orchestrator.submit_task(task_data)
                
                # Get task ID from session
                task_id = str(uuid.uuid4())  # This would be returned by the orchestrator
                
                # Broadcast to WebSocket clients
                await self.connection_manager.broadcast({
                    "type": "task_submitted",
                    "session_id": session_id,
                    "task_name": task_request.name,
                    "timestamp": datetime.now().isoformat()
                })
                
                return TaskResponse(
                    session_id=session_id,
                    task_id=task_id,
                    status="submitted",
                    message="Task submitted successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")
        
        @self.app.get("/tasks")
        async def list_tasks(
            status: Optional[str] = None,
            limit: int = 100,
            offset: int = 0
        ):
            """List tasks with optional filtering"""
            try:
                # Get active tasks
                active_tasks = []
                for task_id, task in self.orchestrator.active_tasks.items():
                    if status and task.status != status:
                        continue
                    
                    active_tasks.append({
                        "task_id": task_id,
                        "name": task.name,
                        "description": task.description,
                        "type": task.type,
                        "priority": task.priority.value,
                        "status": task.status,
                        "assigned_agents": task.assigned_agents,
                        "created_at": task.created_at.isoformat(),
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None
                    })
                
                # Apply pagination
                paginated_tasks = active_tasks[offset:offset + limit]
                
                return {
                    "tasks": paginated_tasks,
                    "total": len(active_tasks),
                    "limit": limit,
                    "offset": offset
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")
        
        @self.app.get("/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """Get detailed status for a specific task"""
            try:
                if task_id not in self.orchestrator.active_tasks:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                
                task = self.orchestrator.active_tasks[task_id]
                
                return {
                    "task_id": task_id,
                    "name": task.name,
                    "description": task.description,
                    "type": task.type,
                    "priority": task.priority.value,
                    "status": task.status,
                    "assigned_agents": task.assigned_agents,
                    "requirements": [cap.value for cap in task.requirements],
                    "payload": task.payload,
                    "constraints": task.constraints,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "results": task.results
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")
        
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            """Cancel a running task"""
            try:
                if task_id not in self.orchestrator.active_tasks:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                
                # Implement task cancellation logic
                task = self.orchestrator.active_tasks[task_id]
                task.status = "cancelled"
                task.completed_at = datetime.now()
                
                # Remove from active tasks
                del self.orchestrator.active_tasks[task_id]
                
                return {"message": f"Task {task_id} cancelled successfully"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")
        
        # ==================== Message Bus Endpoints ====================
        
        @self.app.post("/messages/send")
        async def send_message(sender_id: str, message_request: MessageRequest):
            """Send a message through the message bus"""
            try:
                # Create message
                message = Message(
                    id=str(uuid.uuid4()),
                    sender_id=sender_id,
                    recipient_id=message_request.recipient_id,
                    message_type=MessageType(message_request.message_type),
                    pattern=CommunicationPattern(message_request.pattern),
                    priority=MessagePriority[message_request.priority.upper()],
                    topic=message_request.topic,
                    payload=message_request.payload
                )
                
                if message_request.expires_in_seconds:
                    message.expires_at = datetime.now() + timedelta(seconds=message_request.expires_in_seconds)
                
                # Send message
                success = await self.message_bus.send_message(message)
                
                if success:
                    return {"message_id": message.id, "status": "sent"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to send message")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")
        
        @self.app.post("/messages/broadcast")
        async def broadcast_message(sender_id: str, message_request: MessageRequest):
            """Broadcast a message to all agents"""
            try:
                message = Message(
                    id=str(uuid.uuid4()),
                    sender_id=sender_id,
                    message_type=MessageType(message_request.message_type),
                    pattern=CommunicationPattern.BROADCAST,
                    priority=MessagePriority[message_request.priority.upper()],
                    payload=message_request.payload
                )
                
                success = await self.message_bus.broadcast_message(message)
                
                if success:
                    return {"message_id": message.id, "status": "broadcast"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to broadcast message")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to broadcast message: {str(e)}")
        
        @self.app.get("/messages/stats")
        async def get_message_stats():
            """Get message bus statistics"""
            try:
                stats = await self.message_bus.get_message_stats()
                return stats
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get message stats: {str(e)}")
        
        # ==================== Infrastructure Endpoints ====================
        
        @self.app.get("/infrastructure/status")
        async def get_infrastructure_status():
            """Get infrastructure status"""
            try:
                status = await self.infrastructure.get_system_status()
                return status
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get infrastructure status: {str(e)}")
        
        @self.app.post("/infrastructure/deploy-all")
        async def deploy_all_services():
            """Deploy all agent services"""
            try:
                results = await self.infrastructure.deploy_all_agents()
                
                success_count = sum(1 for success in results.values() if success)
                total_count = len(results)
                
                return {
                    "results": results,
                    "summary": {
                        "total": total_count,
                        "successful": success_count,
                        "failed": total_count - success_count
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to deploy services: {str(e)}")
        
        @self.app.get("/infrastructure/services/{service_name}")
        async def get_service_status(service_name: str):
            """Get status of a specific infrastructure service"""
            try:
                status = await self.infrastructure.get_service_status(service_name)
                if "error" in status:
                    raise HTTPException(status_code=404, detail=status["error"])
                return status
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get service status: {str(e)}")
        
        @self.app.post("/infrastructure/services/{service_name}/control")
        async def control_service(service_name: str, request: ServiceControlRequest):
            """Control infrastructure service"""
            try:
                if request.action == "start":
                    success = await self.infrastructure.deploy_service(service_name)
                elif request.action == "stop":
                    success = await self.infrastructure.stop_service(service_name)
                elif request.action == "restart":
                    success = await self.infrastructure.restart_service(service_name)
                elif request.action == "scale":
                    if request.instances is None:
                        raise HTTPException(status_code=400, detail="instances parameter required for scale action")
                    success = await self.infrastructure.scale_service(service_name, request.instances)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
                
                if success:
                    return {"message": f"Successfully {request.action}ed service {service_name}"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to {request.action} service {service_name}")
                    
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Service control failed: {str(e)}")
        
        # ==================== Configuration Endpoints ====================
        
        @self.app.get("/config")
        async def get_configuration():
            """Get system configuration"""
            return {
                "orchestrator": self.orchestrator.config,
                "message_bus": self.message_bus.config,
                "infrastructure": self.infrastructure.config,
                "api": self.config
            }
        
        @self.app.put("/config")
        async def update_configuration(request: ConfigurationRequest):
            """Update system configuration"""
            try:
                if request.section == "orchestrator":
                    self.orchestrator.config[request.key] = request.value
                elif request.section == "message_bus":
                    self.message_bus.config[request.key] = request.value
                elif request.section == "infrastructure":
                    self.infrastructure.config[request.key] = request.value
                elif request.section == "api":
                    self.config[request.key] = request.value
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown configuration section: {request.section}")
                
                return {"message": f"Configuration updated: {request.section}.{request.key} = {request.value}"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")
        
        # ==================== Analytics Endpoints ====================
        
        @self.app.get("/analytics/performance")
        async def get_performance_analytics(
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
        ):
            """Get performance analytics"""
            try:
                # Default to last 24 hours if no dates provided
                if not end_date:
                    end_date = datetime.now()
                if not start_date:
                    start_date = end_date - timedelta(hours=24)
                
                # Collect performance data
                analytics = {
                    "time_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "agent_performance": {},
                    "task_completion_rate": 0.0,
                    "average_response_time": 0.0,
                    "resource_utilization": {},
                    "message_throughput": 0
                }
                
                # Get agent performance data
                for agent_id, performance in self.orchestrator.agent_performance.items():
                    analytics["agent_performance"][agent_id] = {
                        "success_rate": performance.get("success_rate", 1.0),
                        "response_time": performance.get("response_time", 1.0),
                        "resource_efficiency": performance.get("resource_efficiency", 1.0),
                        "collaboration_score": performance.get("collaboration_score", 0.5)
                    }
                
                # Calculate aggregated metrics
                if analytics["agent_performance"]:
                    analytics["average_success_rate"] = sum(
                        perf["success_rate"] for perf in analytics["agent_performance"].values()
                    ) / len(analytics["agent_performance"])
                    
                    analytics["average_response_time"] = sum(
                        perf["response_time"] for perf in analytics["agent_performance"].values()
                    ) / len(analytics["agent_performance"])
                
                return analytics
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")
        
        @self.app.get("/analytics/coordination")
        async def get_coordination_analytics():
            """Get coordination pattern analytics"""
            try:
                coordination_stats = await self.orchestrator.get_system_status()
                
                return {
                    "coordination_patterns": coordination_stats.get("coordination_patterns", {}),
                    "total_sessions": len(self.orchestrator.active_sessions),
                    "completed_sessions": len(self.orchestrator.completed_tasks),
                    "success_rate": coordination_stats.get("system_metrics", {}).get("avg_success_rate", 0.0),
                    "average_session_duration": 0.0  # Would need to calculate from history
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get coordination analytics: {str(e)}")
        
        # ==================== WebSocket Endpoints ====================
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            client_info = {
                "connected_at": datetime.now(),
                "client_ip": websocket.client.host if websocket.client else "unknown"
            }
            
            await self.connection_manager.connect(websocket, client_info)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(self.config["websocket_heartbeat_interval"])
                    
                    # Send system status update
                    system_status = await self.orchestrator.get_system_status()
                    
                    await self.connection_manager.send_personal_message({
                        "type": "system_status",
                        "data": system_status,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)
        
        @self.app.websocket("/ws/agent/{agent_id}")
        async def agent_websocket(websocket: WebSocket, agent_id: str):
            """WebSocket endpoint for specific agent monitoring"""
            await websocket.accept()
            
            try:
                while True:
                    await asyncio.sleep(10)  # Update every 10 seconds
                    
                    # Get agent-specific status
                    agent_status = await self.orchestrator.get_agent_status(agent_id)
                    if agent_status:
                        await websocket.send_text(json.dumps({
                            "type": "agent_status",
                            "agent_id": agent_id,
                            "data": agent_status,
                            "timestamp": datetime.now().isoformat()
                        }, default=str))
                    
            except WebSocketDisconnect:
                logger.info(f"Agent WebSocket disconnected: {agent_id}")
            except Exception as e:
                logger.error(f"Agent WebSocket error for {agent_id}: {e}")
    
    async def initialize(self):
        """Initialize the unified orchestration API"""
        logger.info("🚀 Initializing Unified Orchestration API...")
        
        # Initialize all components
        await self.orchestrator.initialize()
        await self.message_bus.initialize()
        await self.infrastructure.initialize()
        await self.dashboard.initialize()
        
        # Start background services
        self.background_tasks = [
            asyncio.create_task(self._websocket_broadcaster()),
            asyncio.create_task(self._metrics_updater())
        ]
        
        self.running = True
        logger.info("✅ Unified Orchestration API ready")
    
    async def shutdown(self):
        """Shutdown the unified orchestration API"""
        logger.info("🛑 Shutting down Unified Orchestration API...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Shutdown all components
        await self.orchestrator.shutdown()
        await self.message_bus.shutdown()
        await self.infrastructure.shutdown()
        
        logger.info("✅ Unified Orchestration API shutdown complete")
    
    async def _websocket_broadcaster(self):
        """Background task for broadcasting system updates via WebSocket"""
        
        while self.running:
            try:
                if self.connection_manager.active_connections:
                    # Broadcast system metrics
                    system_status = await self.orchestrator.get_system_status()
                    
                    await self.connection_manager.broadcast({
                        "type": "system_metrics",
                        "data": system_status,
                        "timestamp": datetime.now().isoformat()
                    })
                
                await asyncio.sleep(30)  # Broadcast every 30 seconds
                
            except Exception as e:
                logger.error(f"WebSocket broadcaster error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_updater(self):
        """Background task for updating performance metrics"""
        
        while self.running:
            try:
                # Update orchestration metrics
                # This would typically collect and aggregate metrics from all components
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
                await asyncio.sleep(60)


# ==================== Factory Function ====================

def create_orchestration_api(redis_url: str = "redis://redis:6379") -> UnifiedOrchestrationAPI:
    """Factory function to create the unified orchestration API"""
    return UnifiedOrchestrationAPI(redis_url)


# ==================== FastAPI App Instance ====================

# Create the global API instance
api_instance = create_orchestration_api("redis://redis:6379")
app = api_instance.app

# Add startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    await api_instance.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await api_instance.shutdown()


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "unified_orchestration_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )